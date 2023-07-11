import os

import fire
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.distributed import gather_object

from taide_cp.data import DataModuleForMultipleChoiceQuestion
from taide_cp.models import AutoModelForCausalLM, AutoTokenizer
from taide_cp.training.lightning_module import LightningModuleX
from taide_cp.utils import write_json
from taide_cp.utils.scripting import get_logger, get_trainer


class LightningModuleForMultipleChoiceQuestion(LightningModuleX):
    def __init__(self, model_path: str) -> None:
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.example_input_ids = None
        self.example_past_kv = None

        self.total = 0
        self.correct = 0
        self.entropy_list = []

        self.results = {}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        batch_size = batch['example_encoding']['input_ids'].size(0)
        if self.example_past_kv is None or self.example_past_kv[0][0].size(0) != batch_size:
            self.example_past_kv = self.model(**batch['example_encoding'], use_cache=True).past_key_values
        
        probs = []
        for question_choice_encoding, choice_target, choice_index, choice_index_padding_mask in zip(
            batch['question_choice_encodings'],
            batch['choice_targets'],
            batch['choice_indices'],
            batch['choice_index_padding_masks'],
        ):
            logits = self.model(**question_choice_encoding, past_key_values=self.example_past_kv, use_cache=False).logits
            p = torch.log_softmax(logits, dim=-1)
            p = p.gather(1, choice_index.unsqueeze(-1).repeat(1, 1, p.size(-1)))
            p = p.gather(2, choice_target).masked_fill(choice_index_padding_mask.unsqueeze(-1), 0.0).squeeze()
            p = p.sum(-1).div(choice_index_padding_mask.logical_not().count_nonzero(-1)).exp()
            probs.append(p)
        
        probs = torch.stack(probs, dim=1)
        probs = probs / probs.sum(-1).unsqueeze(-1)
        entropy = -probs.log().mul(probs).sum(-1)
        
        correct = 0
        total = 0
        entropy_list = []
        for i, example, question, choices, answer, p, e in zip(
            batch['id'],
            batch['example'],
            batch['question'],
            batch['choices'],
            batch['answer'],
            probs,
            entropy
        ):
            e = e.item()
            index = p.argmax().item()
            entropy_list.append(e)
            
            self.results[i] = {
                'example': example,
                'question': question,
                'choices': choices,
                'answer': answer['text'],
                'prediction': {
                    'text': choices[index],
                    'probs': p.tolist(),
                    'entropy': e,
                }
            }

            correct += 1 if index == answer['index'] else 0
            total += 1

        self.correct += correct
        self.total += total
        self.entropy_list += entropy_list

        log_kwargs = dict(
            prog_bar=True,
            logger=False,
            on_step=True,
            sync_dist=True,
        )
        self.log('correct', self.correct, reduce_fx='sum', **log_kwargs)
        self.log('total', self.total, reduce_fx='sum', **log_kwargs)
        self.log('entropy', sum(self.entropy_list) / len(self.entropy_list), **log_kwargs)
        self.log('accuracy', self.correct / self.total, **log_kwargs)

    def on_test_epoch_end(self) -> None:
        if self.trainer.world_size > 1:
            object_gather_list = [None] * self.trainer.world_size if self.global_rank == 0 else None
            gather_object(self.results, object_gather_list)

        if self.global_rank == 0:
            for r in object_gather_list:
                self.results |= r
            
            keys = sorted(self.results.keys())
            self.results = [self.results[k] for k in keys]

            correct = 0
            entropy = 0
            total = len(self.results)
            for x in self.results:
                if x['answer'] == x['prediction']['text']:
                    correct += 1
                entropy += x['prediction']['entropy']
            
            self.log('Correct', correct)
            self.log('Total', total)
            self.log('Entropy', entropy / total)
            self.log('Accuracy', correct / total)

def main(
    model_path: str,
    data_path: str,
    k_shot: int,
    batch_size: int = 1,
    num_datapoints: int | None = None,
    save_dir: str | None = 'logs/eval',
    name: str | None = None,
    version: str | None = None,
):
    model = LightningModuleForMultipleChoiceQuestion(model_path)
    datamodule = DataModuleForMultipleChoiceQuestion(
        tokenizer=model.tokenizer,
        k_shot=k_shot,
        data_path=data_path,
        test_split_size=num_datapoints,
        test_batch_size=batch_size,
    )

    trainer = get_trainer(
        logger=get_logger(
            logger_type='csv',
            save_dir=save_dir,
            name=name,
            version=version,
        ),
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule)
    if model.is_global_zero:
        write_json(f'{trainer.logger.log_dir}/results.json', model.results)


if __name__ == '__main__':
    fire.Fire(main)
