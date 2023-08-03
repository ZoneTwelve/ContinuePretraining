import os
from collections import defaultdict, OrderedDict
from typing import Dict, List

import fire
from taide_cp.utils.scripting.decorators import entry_point
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
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

        self.example_past_kv = None

        self.total = 0
        self.correct = 0
        self.entropy_sum = 0.0

        self.results = {}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        # The example cache does not work on the MPT model currently.
        batch_size = batch['question_choice_inputs'][0]['encoding'][
            'input_ids'].size(0)
        if batch['example_encoding'] is not None and (
                self.example_past_kv is None
                or self.example_past_kv[0][0].size(0) != batch_size):
            self.example_past_kv = self.model(**batch['example_encoding'],
                                              use_cache=True).past_key_values

        posterior = []
        prior = []
        for question_choice_input, choice_input in zip(
                batch['question_choice_inputs'], batch['choice_inputs']):
            p = self.model(**question_choice_input['encoding'],
                           past_key_values=self.example_past_kv,
                           use_cache=False).logits
            p = torch.log_softmax(p, dim=-1)
            p = p.gather(
                1, question_choice_input['choice_index'].unsqueeze(-1).repeat(
                    1, 1, p.size(-1)))
            p = p.gather(2, question_choice_input['choice_target'])
            p = p.masked_fill(
                question_choice_input['choice_index_padding_mask'].unsqueeze(
                    -1), 0.0).squeeze()
            p = p.sum(-1).div(
                question_choice_input['choice_index_padding_mask'].logical_not(
                ).count_nonzero(-1)).exp()
            posterior.append(p)

            p = self.model(**choice_input['encoding'], use_cache=False).logits
            p = torch.log_softmax(p, dim=-1)
            p = p.gather(
                1,
                choice_input['index'].unsqueeze(-1).repeat(1, 1, p.size(-1)))
            p = p.gather(2, choice_input['target']).masked_fill(
                choice_input['index_padding_mask'].unsqueeze(-1),
                0.0).squeeze()
            p = p.sum(-1).div(
                choice_input['index_padding_mask'].logical_not().count_nonzero(
                    -1)).exp()
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
                'lv1_domain': x.get('lv1_domain', ''),
                'lv2_domain': x.get('lv2_domain', ''),
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

            self.correct += 1 if posterior[i].argmax(
            ) == x['answer']['index'] else 0
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
            object_gather_list = [
                None
            ] * self.trainer.world_size if self.global_rank == 0 else None
            gather_object(self.results, object_gather_list)

        if self.global_rank == 0:
            for r in object_gather_list:
                self.results |= r

            keys = sorted(self.results.keys())
            self.results = [self.results[k] for k in keys]
            metrics: Dict[str, float] = self.__calc_metrics(self.results)

            
            if 'lv1_domain' in self.results[0]:  # have domain info
                metrics['avg_acc_lv1/information_gain'] = 0
                metrics['avg_acc_lv2/information_gain'] = 0
                metrics['avg_acc_lv1/raw'] = 0
                metrics['avg_acc_lv2/raw'] = 0
                domain_result_lv1: Dict[str, list] = defaultdict(lambda: [])
                domain_result_lv2: Dict[str, list] = defaultdict(lambda: [])

                # groupby
                for r in self.results:
                    if not r['lv1_domain']:
                        continue
                    domain_result_lv1[r['lv1_domain']].append(r)
                    domain_result_lv2[
                        f'{r["lv1_domain"]}-{r["lv2_domain"]}'].append(r)

                for dom, result in domain_result_lv1.items():
                    dom_metrics = self.__calc_metrics(result)
                    for k, v in dom_metrics.items():
                        metrics[f'{dom}/{k}'] = v
                    metrics['avg_acc_lv1/information_gain'] += metrics[f'{dom}/acc/information_gain']
                    metrics['avg_acc_lv1/raw'] += metrics[f'{dom}/acc/raw']
                
                for dom, result in domain_result_lv2.items():
                    dom_metrics = self.__calc_metrics(result)
                    for k, v in dom_metrics.items():
                        metrics[f'{dom}/{k}'] = v
                    metrics['avg_acc_lv2/information_gain'] += metrics[f'{dom}/acc/information_gain']
                    metrics['avg_acc_lv2/raw'] += metrics[f'{dom}/acc/raw'] 

                if domain_result_lv1:
                    metrics['avg_acc_lv1/information_gain'] /= len(domain_result_lv1)
                    metrics['avg_acc_lv1/raw'] /= len(domain_result_lv1)
                
                if domain_result_lv2:
                    metrics['avg_acc_lv2/information_gain'] /= len(domain_result_lv2)
                    metrics['avg_acc_lv2/raw'] /= len(domain_result_lv2)
            
            metrics = OrderedDict(sorted(metrics.items(), key=lambda x: x[0]))
            self.log_dict(metrics, rank_zero_only=True)
            self.metrics = metrics

    @staticmethod
    def __calc_metrics(result: Dict[str, List]) -> Dict[str, float]:
        correct_raw = 0
        correct_ig = 0
        entropy = 0
        total = len(result)
        
        for x in result:
            if x['answer'] == x['prediction']['raw']['answer']:
                correct_raw += 1

            if x['answer'] == x['prediction']['information_gain']['answer']:
                correct_ig += 1

            entropy += x['prediction']['entropy']

        return {
            'total': total,
            'entropy': entropy / total,
            'correct/raw': correct_raw,
            'acc/raw': correct_raw / total,
            'correct/information_gain': correct_ig,
            'acc/information_gain': correct_ig / total,
        }

@entry_point(
    get_trainer,
    get_logger
)
def main(
    model_path: str,
    data_path: str,
    k_shot: int,
    batch_size: int = 1,
    convert_to_chs: bool = False,
    num_datapoints: int | None = None,
    save_dir: str | None = 'logs/eval',
    **kwargs
):
    kwargs['name'] = kwargs.pop('name', os.path.join('mcq', f'{k_shot}_shot'))
    kwargs['version'] = kwargs.pop('version', os.path.basename(model_path) + f'_bs{batch_size}')

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
            **kwargs
        ),
        enable_checkpointing=False,
        **kwargs
    )

    trainer.test(model, datamodule)

    if trainer.is_global_zero:
        path = os.path.join(trainer.logger.log_dir, 'results.json')
        result = {'metrics': model.metrics, 'results': model.results}
        write_json(path, result)
        print(f'The result is dumped to: {path}')


if __name__ == '__main__':
    fire.Fire(main)
