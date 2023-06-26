import os
from glob import glob
from typing import Optional

import fire
import lightning as L
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import taide_cp.models
from taide_cp.data import DataCollatorForEvaluation
from taide_cp.metrics import Perplexity
from taide_cp.utils import DatasetsContextManager
from taide_cp.utils.scripting import *


def normalize_dataset(dataset: Dataset):
    dataset = dataset.rename_columns({o: o.lower() for o in dataset.column_names})
    column_mapping = {'instruction': 'prompt', 'output': 'response'}
    column_mapping = {k: v for k, v in column_mapping.items() if k in dataset.column_names}
    dataset = dataset.rename_columns(column_mapping)
    if 'input' not in dataset.column_names:
        dataset = dataset.add_column('input', [None] * len(dataset))
    dataset = dataset.select_columns(['prompt', 'input', 'response'])
    return dataset

@DatasetsContextManager()
def load_datasets(data_path: str) -> Dataset:
    paths = []
    if os.path.isfile(data_path):
        paths = [data_path]
    else:
        paths = glob(os.path.join(data_path, '**/*.*'), recursive=True)

    datasets = []
    for p in paths:
        x = load_dataset('csv', data_files=p)['train']
        x = normalize_dataset(x)
        datasets.append(x)

    return concatenate_datasets(datasets)


class LightningModuleForEvaluation(L.LightningModule):
    def __init__(
        self,
        model_path: str
    ) -> None:
        super().__init__()

        self.model_path = model_path

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype='auto', low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        self.perplexity = Perplexity(ignore_index=-100)

    def test_step(self, batch, batch_idx):
        x = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],        
        )
        self.perplexity.update(x.loss, batch['labels'], by_loss=True)
        self.log('Perplexity/Test', self.perplexity, batch_size=batch['input_ids'].size(0), sync_dist=True)

@entry_point(
    get_wandb_logger
)
def main(
    model_path: str | None = None,
    ckpt_path: str | None = None,
    data_path: str = 'data/sft/raw',
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs,
):
    if model_path:
        model = LightningModuleForEvaluation(model_path)
    else:
        checkpoint = torch.load(ckpt_path, 'cpu')
        model = LightningModuleForEvaluation(checkpoint['hyper_parameters']['model_path'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    dataset = load_datasets(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=DataCollatorForEvaluation(model.tokenizer))
    
    trainer = L.Trainer(
        num_nodes=int(os.environ.get('SLURM_NNODES', '0')) or 'auto',
        logger=get_wandb_logger(**kwargs),
        enable_checkpointing=False,
    )

    trainer.test(model, dataloader)
    trainer.logger.experiment.finish()

if __name__ == '__main__':
    fire.Fire(main)
