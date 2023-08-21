import glob
import os
from typing import Any, Dict, List, Optional, Union

from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

from ...utils import DatasetsContextManager
from ..datamodule import LightningDataModuleX, StageType
from .datacollator_for_pre_training import DataCollatorForPreTraining


@DatasetsContextManager()
def load_datasets(data_path: str) -> Dataset:
    datasets = []
    paths = []
    if os.path.isdir(data_path):
        paths = glob.glob(os.path.join(data_path, '**/*.*'), recursive=True)
        paths = list(filter(lambda p: os.path.isfile(p), paths))
    else:
        paths = [data_path]

    progress = tqdm(total=len(paths), desc='Loading Files', leave=False)
    for p in paths:
        if os.path.isfile(p):
            x = load_dataset('json', data_files=p)['train']
        else:
            x = load_dataset(p)['train']
        datasets.append(x)
        progress.update()

    return concatenate_datasets(datasets)

def tokenize(batch: List[str], tokenizer: PreTrainedTokenizer):
    batch = tokenizer(batch, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False, verbose=False)
    for input_ids in batch['input_ids']:
        input_ids[:] = [
            tokenizer.bos_token_id,
            *input_ids,
            tokenizer.eos_token_id
        ]
    return batch

def rearrange_datapoints(
    batch: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizer,
    sequence_length: int,
):
    datapoints = []
    
    input_ids = []
    for x in batch['input_ids']:
        input_ids += x
        while len(input_ids) >= sequence_length:
            datapoint = input_ids[:sequence_length]
            datapoints.append(datapoint)
            input_ids[:sequence_length] = []

    if input_ids:
        paddings = [-1] * (sequence_length - len(input_ids))
        datapoint = paddings + input_ids if tokenizer.padding_side == 'left' else input_ids + paddings
        datapoints.append(datapoint)
    
    batch['input_ids'] = datapoints
    return batch


class DataModuleForPreTraining(LightningDataModuleX):
    datacollator_cls = DataCollatorForPreTraining

    @property
    def tokenized_dataset_path(self):
        return os.path.join(self.dataset_path, 'tokenized')

    @property
    def rearranged_dataset_path(self):
        return os.path.join(self.dataset_path, f'rearranged/{self.sequence_length}')

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        *,
        data_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        train_batch_size: int = 1,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        predict_batch_size: Optional[int] = None,
        val_split_size: Optional[Union[int, float]] = None,
        test_split_size: Optional[Union[int, float]] = None,
        predict_split_size: Optional[Union[int, float]] = None,
        split_seed: Optional[int] = 42,
        num_workers: int = 1,
        pin_memory: bool = True,
        num_proc: Optional[int] = None,
        datacollator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            tokenizer,
            data_path=data_path,
            dataset_path=dataset_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            pred_batch_size=predict_batch_size,
            val_split_size=val_split_size,
            test_split_size=test_split_size,
            pred_split_size=predict_split_size,
            split_seed=split_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_proc=num_proc,
            datacollator_kwargs=datacollator_kwargs
        )
        self.sequence_length = sequence_length
 
    @DatasetsContextManager(progress_bar=True)
    def prepare_data(self) -> None:
        if self.is_dataset(self.rearranged_dataset_path):
            rearranged_dataset = load_from_disk(self.rearranged_dataset_path)
        else:
            if self.is_dataset(self.tokenized_dataset_path):
                tokenized_dataset = load_from_disk(self.tokenized_dataset_path)
            else:
                dataset = load_datasets(self.data_path)
                tokenized_dataset = dataset.map(
                    tokenize,
                    fn_kwargs=dict(tokenizer=self.tokenizer),
                    input_columns='text',
                    remove_columns=dataset.column_names,
                    batched=True,
                    num_proc=self.num_proc,
                )
                tokenized_dataset.save_to_disk(self.tokenized_dataset_path)

            rearranged_dataset = tokenized_dataset.map(
                rearrange_datapoints,
                fn_kwargs=dict(
                    tokenizer=self.tokenizer,
                    sequence_length=self.sequence_length,
                ),
                batched=True,
                num_proc=self.num_proc,
            )

            rearranged_dataset.save_to_disk(self.rearranged_dataset_path)
    
    def setup(self, stage: Optional[StageType] = None) -> None:
        self.dataset = load_from_disk(self.rearranged_dataset_path)
        self.dataset = self.split(self.dataset)

    def count_tokens(self, tokenized_dataset: Dataset | None = None):
        if tokenized_dataset is None:
            tokenized_dataset = load_from_disk(self.tokenized_dataset_path)

        dataset = tokenized_dataset.map(
            lambda x: {'num_tokens': len(x['input_ids'])},
            remove_columns=tokenized_dataset.column_names,
            num_proc=self.num_proc
        )
        num_tokens = sum(l for l in dataset['num_tokens'])
        return num_tokens
