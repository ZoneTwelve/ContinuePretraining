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
    def __init__(self, model_path: str, normalize_by_uncond: bool) -> None:
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.normalize_by_uncond = normalize_by_uncond

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
        for question_choice_input, answer_input in zip(batch['question_choice_inputs'], batch['answer_inputs']):
            p = self.model(**question_choice_input['encoding'], past_key_values=self.example_past_kv, use_cache=False).logits
            p = torch.log_softmax(p, dim=-1)
            p = p.gather(1, question_choice_input['choice_index'].unsqueeze(-1).repeat(1, 1, p.size(-1)))
            p = p.gather(2, question_choice_input['choice_target'])
            p = p.masked_fill(question_choice_input['choice_index_padding_mask'].unsqueeze(-1), 0.0).squeeze()
            p = p.sum(-1).div(question_choice_input['choice_index_padding_mask'].logical_not().count_nonzero(-1)).exp()

            if self.normalize_by_uncond:
                ap = self.model(**answer_input['encoding'], use_cache=False).logits
                ap = torch.log_softmax(ap, dim=-1)
                ap = ap.gather(1, answer_input['index'].unsqueeze(-1).repeat(1, 1, ap.size(-1)))
                ap = ap.gather(2, answer_input['target']).masked_fill(answer_input['index_padding_mask'].unsqueeze(-1), 0.0).squeeze()
                ap = ap.sum(-1).div(answer_input['index_padding_mask'].logical_not().count_nonzero(-1)).exp()
                p /= ap
            
            probs.append(p)
        
        probs = torch.stack(probs, dim=1)
        probs = probs / probs.sum(-1).unsqueeze(-1)
        entropy = -probs.log().mul(probs).sum(-1)
        
        correct = 0
        total = 0
        entropy_list = []
        for x, p, e in zip(batch['data'], probs, entropy):
            e = e.item()
            index = p.argmax().item()
            entropy_list.append(e)
            
            self.results[x['id']] = {
                'example': x['example'],
                'question': x['question'],
                'choices': x['choices'],
                'answer': x['answer']['text'],
                'prediction': {
                    'text': x['choices'][index],
                    'probs': p.tolist(),
                    'entropy': e,
                }
            }

            correct += 1 if index == x['answer']['index'] else 0
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
    normalize_by_uncond: bool = False,
    convert_to_chs: bool = False,
    num_datapoints: int | None = None,
    save_dir: str | None = 'logs/eval',
    name: str | None = None,
    version: str | None = None,
):
    model = LightningModuleForMultipleChoiceQuestion(model_path, normalize_by_uncond=normalize_by_uncond)
    datamodule = DataModuleForMultipleChoiceQuestion(
        tokenizer=model.tokenizer,
        k_shot=k_shot,
        convert_to_chs=convert_to_chs,
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
        path = os.path.join(trainer.logger.log_dir, 'results.json')
        write_json(path, model.results)
        print(f'The result is dumped to: {path}')

if __name__ == '__main__':
    fire.Fire(main)
