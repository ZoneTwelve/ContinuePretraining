import json
from pathlib import Path
from typing import Dict, Optional, cast

import fire
import lightning as L
from datasets import disable_caching, disable_progress_bar, load_dataset
from torch import Tensor
from torch.distributed import gather_object
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)

from taide_cp.lightning import DeepSpeedStrategy
from taide_cp.utils.data import DataCollator
from tc_scorer.inference import TwCnStyleEvaluator


class EvaluationModule(L.LightningModule):
    def __init__(
        self,
        model_path: str,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.bos_token})

        assert self.tokenizer.pad_token is not None

        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        self.results = {}
        self.score = None

    def configure_sharded_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model = cast(PreTrainedModel, self.model)
        self.evaluator = TwCnStyleEvaluator('aqweteddy/bertbase-twcnstyle-entdebias')

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            generation_config=GenerationConfig(
                max_new_tokens=128,
            ),
            use_cache=True,
            synced_gpus=True
        )
        x = x[:, batch['input_ids'].size(1):]
        batch_text = self.tokenizer.batch_decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        batch_score = self.evaluator(batch_text, batch_size=len(x))
        for title, text, score in zip(batch['title'], batch_text, batch_score):
            self.results[title] = (text, score.item())
    
    def on_validation_epoch_end(self) -> None:
        object_gather_list = [None] * self.trainer.world_size if self.global_rank == 0 else None
        gather_object(self.results, object_gather_list)

        if self.global_rank == 0:
            for r in object_gather_list:
                self.results |= r

            total_score = 0
            for _, score in self.results.values():
                total_score += score
            self.score = total_score / len(self.results)

            self.log('score', self.score, rank_zero_only=True)

class EvaluationDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

    def __call__(self, batch) -> Dict[str, Tensor]:
        batch = self.convert_list_of_dict_to_dict_of_list(batch)
        new_batch = {
            'title': batch['title']
        }
        new_batch |= self.tokenizer(
            batch['title'],
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True
        )
        return new_batch


def main(
    model_path: str,
    data_path: str,
    output_path: str = 'outputs/tc_score/',
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    num_samples: Optional[int] = 1000,
):
    disable_caching()
    disable_progress_bar()
    
    model = EvaluationModule(model_path)

    dataset = load_dataset('json', data_files=data_path)['train']
    num_samples = num_samples or dataset.num_rows
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    datacollator = EvaluationDataCollator(model.tokenizer)
    dataloaders = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, collate_fn=datacollator)

    trainer = L.Trainer(
        strategy=DeepSpeedStrategy(),
        precision='16-mixed',
        logger=False,
    )
    outputs = trainer.validate(model, dataloaders=dataloaders)
    if model.global_rank == 0:
        score = outputs[0]['score']
        print(Path(model_path).name, score)
        output_path: Path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path.joinpath(Path(model_path).name + '.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            results = [{'title': title, 'generated': text, 'score': score} for title, (text, score) in model.results.items()]
            json.dump({'score': score, 'results': results}, f, ensure_ascii=False, indent=True)

if __name__ == '__main__':
    fire.Fire(main)
