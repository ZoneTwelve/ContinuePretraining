import glob
import json
import os

from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from ...utils import DatasetsContextManager
from ..datamodule import DataModuleConfig, LightningDataModuleX, StageType
from .datacollator_for_pre_training import DataCollatorForPreTraining


def generator(paths: list):
    progress = tqdm(total=len(paths), desc='Loading Files', leave=False)
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for l in f:
                x = json.loads(l)
                yield {'text': x['text']}
        progress.update()


@DatasetsContextManager()
def load_datasets(path: str, num_proc: int = None) -> Dataset:
    paths = []
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, '**/*.*'), recursive=True)
        paths = list(filter(lambda p: os.path.isfile(p), paths))
        dataset = Dataset.from_generator(generator, gen_kwargs=dict(paths=paths), num_proc=num_proc)
    elif os.path.isfile(path):
        paths = [path]
        dataset = Dataset.from_generator(generator, gen_kwargs=dict(paths=paths), num_proc=num_proc)
    else:
        dataset = load_dataset(path)['train']
        dataset = dataset.select_columns('text')
    return dataset


def tokenize(batch: list[str], tokenizer: PreTrainedTokenizer):
    batch = tokenizer(
        [x for x in batch if x],
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False
    )
    for input_ids in batch['input_ids']:
        input_ids[:] = [
            tokenizer.bos_token_id,
            *input_ids,
            tokenizer.eos_token_id
        ]
    return batch


def rearrange_datapoints(
    batch: dict[str, list[int]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
    datapoints = []
    
    input_ids = []
    for x in batch['input_ids']:
        input_ids += x
        while len(input_ids) >= max_length:
            datapoint = input_ids[:max_length]
            datapoints.append(datapoint)
            input_ids[:max_length] = []

    if input_ids:
        paddings = [-1] * (max_length - len(input_ids))
        datapoint = paddings + input_ids if tokenizer.padding_side == 'left' else input_ids + paddings
        datapoints.append(datapoint)
    
    batch['input_ids'] = datapoints
    return batch


class DataModuleForPreTrainingConfig(DataModuleConfig):
    max_length: int | None = None


class DataModuleForPreTraining(LightningDataModuleX):
    config: DataModuleForPreTrainingConfig

    @property
    def tokenized_dataset_path(self):
        return os.path.join(self.config.dataset_path, 'tokenized')

    @property
    def rearranged_dataset_path(self):
        return os.path.join(self.config.dataset_path, f'rearranged/{self.config.max_length}')

    def __init__(self, config: DataModuleForPreTrainingConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(config)
        
        self.tokenizer = tokenizer
        self.datacollator = DataCollatorForPreTraining(tokenizer)
 
    @DatasetsContextManager(progress_bar=True)
    def prepare_data(self) -> None:
        if self.config.max_length is not None and self.is_dataset(self.rearranged_dataset_path):
            rearranged_dataset = load_from_disk(self.rearranged_dataset_path)
        else:
            if self.is_dataset(self.tokenized_dataset_path):
                tokenized_dataset = load_from_disk(self.tokenized_dataset_path)
            else:
                dataset: Dataset = concatenate_datasets([load_datasets(p) for p in self.config.data_path])

                tokenized_dataset = dataset.map(
                    tokenize,
                    fn_kwargs=dict(tokenizer=self.tokenizer),
                    input_columns='text',
                    remove_columns=dataset.column_names,
                    batched=True,
                    num_proc=self.config.num_proc,
                )
                tokenized_dataset.save_to_disk(self.tokenized_dataset_path)

            if self.config.max_length is not None:
                rearranged_dataset = tokenized_dataset.map(
                    rearrange_datapoints,
                    fn_kwargs=dict(
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_length,
                    ),
                    batched=True,
                    num_proc=self.config.num_proc,
                )

                rearranged_dataset.save_to_disk(self.rearranged_dataset_path)
    
    def setup(self, stage: StageType | None = None) -> None:
        self.dataset = load_from_disk(self.rearranged_dataset_path)
        self.dataset = self.split(self.dataset)

    @DatasetsContextManager(progress_bar=True)
    def count_tokens(self, tokenized_dataset: Dataset | None = None):
        if tokenized_dataset is None:
            tokenized_dataset = load_from_disk(self.tokenized_dataset_path)

        dataset = tokenized_dataset.map(
            lambda x: {'num_tokens': len(x['input_ids'])},
            remove_columns=tokenized_dataset.column_names,
            num_proc=self.config.num_proc
        )
        num_tokens = sum(l for l in dataset['num_tokens'])
        return num_tokens
