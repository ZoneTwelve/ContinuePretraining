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
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.example_past_kv = None

        self.total = 0
        self.correct = 0
        self.entropy_sum = 0.0

        self.results = {}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        # The example cache does not work on the MPT model currently.
        batch_size = batch['question_choice_inputs'][0]['encoding']['input_ids'].size(0)
        if batch['example_encoding'] is not None and (self.example_past_kv is None or self.example_past_kv[0][0].size(0) != batch_size):
            self.example_past_kv = self.model(**batch['example_encoding'], use_cache=True).past_key_values

        posterior = []
        prior = []
        for question_choice_input, choice_input in zip(batch['question_choice_inputs'], batch['choice_inputs']):
            p = self.model(**question_choice_input['encoding'], past_key_values=self.example_past_kv, use_cache=False).logits
            p = torch.log_softmax(p, dim=-1)
            p = p.gather(1, question_choice_input['choice_index'].unsqueeze(-1).repeat(1, 1, p.size(-1)))
            p = p.gather(2, question_choice_input['choice_target'])
            p = p.masked_fill(question_choice_input['choice_index_padding_mask'].unsqueeze(-1), 0.0).squeeze()
            p = p.sum(-1).div(question_choice_input['choice_index_padding_mask'].logical_not().count_nonzero(-1)).exp()
            posterior.append(p)
        
            p = self.model(**choice_input['encoding'], use_cache=False).logits
            p = torch.log_softmax(p, dim=-1)
            p = p.gather(1, choice_input['index'].unsqueeze(-1).repeat(1, 1, p.size(-1)))
            p = p.gather(2, choice_input['target']).masked_fill(choice_input['index_padding_mask'].unsqueeze(-1), 0.0).squeeze()
            p = p.sum(-1).div(choice_input['index_padding_mask'].logical_not().count_nonzero(-1)).exp()
            prior.append(p)
        
        posterior = torch.stack(posterior, dim=1)
        prior = torch.stack(prior, dim=1)
        information_gain = posterior.log() - prior.log()

        entropy = posterior / posterior.sum(-1).unsqueeze(-1)
        entropy = -entropy.log().mul(entropy).sum(-1)
        
        for i, x in enumerate(batch['data']):            
            self.results[x['id']] = {
                'example': x['example'],
                'question': x['question'],
                'choices': x['choices'],
                'answer': x['answer']['text'],
                'prediction': {
                    'raw': {
                        'answer': x['choices'][posterior[i].argmax()],
                        'score': posterior[i].tolist()
                    },
                    'information_gain': {
                        'answer': x['choices'][information_gain[i].argmax()],
                        'score': information_gain[i].tolist()
                    },
                    'entropy': entropy[i].item()
                }
            }

            self.correct += 1 if posterior[i].argmax() == x['answer']['index'] else 0
            self.total += 1

        self.entropy_sum += entropy.sum().item()

        log_kwargs = dict(
            prog_bar=True,
            logger=False,
            on_step=True,
            sync_dist=True,
        )
        self.log('correct', self.correct, reduce_fx='sum', **log_kwargs)
        self.log('total', self.total, reduce_fx='sum', **log_kwargs)
        self.log('entropy', self.entropy_sum / self.total, **log_kwargs)
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

            correct_raw = 0
            correct_ig = 0
            entropy = 0
            total = len(self.results)
            for x in self.results:
                if x['answer'] == x['prediction']['raw']['answer']:
                    correct_raw += 1

                if x['answer'] == x['prediction']['information_gain']['answer']:
                    correct_ig += 1
                    
                entropy += x['prediction']['entropy']
            
            self.log('Total', total)
            self.log('Entropy', entropy / total)

            self.log('Correct/Raw', correct_raw)
            self.log('Accuracy/Raw', correct_raw / total)
            self.log('Correct/InformationGain', correct_ig)
            self.log('Accuracy/InformationGain', correct_ig / total)


def main(
    model_path: str,
    data_path: str,
    k_shot: int,
    batch_size: int = 1,
    convert_to_chs: bool = False,
    num_datapoints: int | None = None,
    save_dir: str | None = 'logs/eval',
    name: str | None = None,
    version: str | None = None,
):
    model = LightningModuleForMultipleChoiceQuestion(model_path)
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

    if trainer.is_global_zero:
        path = os.path.join(trainer.logger.log_dir, 'results.json')
        write_json(path, model.results)
        print(f'The result is dumped to: {path}')


if __name__ == '__main__':
    fire.Fire(main)
