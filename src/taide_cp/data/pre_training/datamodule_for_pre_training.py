import logging
import math
import random

from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..datamodule import DataModule
from .datacollator_for_pre_training import DataCollatorForPreTraining
from .datamodule_for_pre_training_config import (
    ConcatMethod, DataModuleForPreTrainingConfig)

logger = logging.getLogger(__name__)


class DataModuleForPreTraining(DataModule):
    config: DataModuleForPreTrainingConfig
    datacollator_class = DataCollatorForPreTraining

    def __init__(self, config: DataModuleForPreTrainingConfig) -> None:
        super().__init__(config)

    def _tokenize(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Tokenize')

        if self.config.shuffle_before_tokenization:
            dataset_dict = dataset_dict.shuffle(seed=42)
            dataset_dict = dataset_dict.flatten_indices(num_proc=self.config.num_proc)
        
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns=True,
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
            desc='Tokenize'
        )
        return dataset_dict
    
    def _truncate(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Truncate')

        dataset_dict = dataset_dict.map(
            _truncate,
            batched=True,
            fn_kwargs=dict(
                max_length=self.config.max_length,
                stride=self.config.stride
            ),
            num_proc=self.config.num_proc,
            desc='Truncate'
        )
        return dataset_dict
    
    def _partition_by_length(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        eq_indices = []
        lt_indices = []
        progress = tqdm(total=len(dataset), desc='Partition by length')
        i = 0
        for batch in dataset.select_columns('length').iter(1000):
            batch_size = len(batch['length'])
            for length in batch['length']:
                indices = eq_indices if length == self.config.max_length else lt_indices
                indices.append(i)
                i += 1 
            progress.update(batch_size)
        return dataset.select(eq_indices), dataset.select(lt_indices)
    
    def _concat_and_truncate(self, dataset_dict: DatasetDict) -> DatasetDict:
        logger.info('Concat and truncate')

        for split, dataset in dataset_dict.items():
            eq_dataset, lt_dataset = self._partition_by_length(dataset)
            lt_dataset = lt_dataset.flatten_indices(num_proc=self.config.num_proc)
            logger.info('Sort by source')
            lt_dataset = lt_dataset.sort('source')
            logger.info('Sorted')
            lt_dataset = lt_dataset.map(
                _concat_and_truncate,
                batched=True,
                batch_size=100000,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Concat and truncate'
            )
            dataset_dict[split] = concatenate_datasets([eq_dataset, lt_dataset])
        return dataset_dict

    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self._tokenize(dataset_dict)

        if self.config.max_length is not None:
            dataset_dict = self._truncate(dataset_dict)

        if self.config.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            dataset_dict = self._concat_and_truncate(dataset_dict)
        
        return dataset_dict
    
    def _partition_by_source(self, dataset: Dataset) -> dict[str, Dataset]:
        source_to_indices = {}
        progress = tqdm(total=len(dataset), desc='Partition by source')
        i = 0
        for batch in dataset.select_columns('source').iter(1000):
            for source in batch['source']:
                indices = source_to_indices.setdefault(source, [])
                indices.append(i)
                i += 1 
                progress.update()
        return {source: dataset.select(indices) for source, indices in source_to_indices.items()}
    
    def sample_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        if all(x == 1.0 for x in self.config.sample_rate.values()):
            return dataset_dict

        r = random.Random(42)
        unused_sample_rate = self.config.sample_rate.copy()
        for k, dataset in dataset_dict.items():
            if k == 'train':
                dataset_list = []
                for source, dataset in self._partition_by_source(dataset).items():
                    sample_rate = self.config.sample_rate.get(source, 1.0)
                    unused_sample_rate.pop(source, None)
                    decimal, integer = math.modf(sample_rate)
                    dataset_list += [dataset] * int(integer)
                    if decimal > 0.0:
                        n = len(dataset)
                        indices = r.sample(list(range(n)), k=int(n * decimal))
                        dataset_list += [dataset.select(indices)]
                dataset_dict[k] = concatenate_datasets(dataset_list)
        
        if len(unused_sample_rate) > 0:
            logger.warn(f'Some sources specified by `sample_rate` are not found in the dataset:\n {unused_sample_rate}')

        return dataset_dict
    
    def post_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = super().post_process_data(dataset_dict)
        dataset_dict = self.sample_data(dataset_dict)
        return dataset_dict


def _tokenize(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    batch_text = [text for text in batch['text'] if text]
    batch_source = [source for text, source in zip(batch['text'], batch['source']) if text] if 'source' in batch else [None] * len(batch_text)
    batch_input_ids: list[list[int]] = tokenizer(
        batch_text,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False
    )['input_ids']

    batch_length = []
    for input_ids in batch_input_ids:
        input_ids.insert(0, tokenizer.bos_token_id)
        input_ids.append(tokenizer.eos_token_id)
        batch_length.append(len(input_ids))
    
    return {
        'input_ids': batch_input_ids,
        'source': batch_source,
        'length': batch_length
    }


def _truncate(batch: dict[str, list[int]], max_length: int, stride: int):
    batch_input_ids = []
    batch_source = []
    batch_length = []
    for source, input_ids in zip(batch['source'], batch['input_ids']):
        for i in range(0, len(input_ids), stride):
            t = input_ids[i:i + max_length]
            batch_input_ids.append(t)
            batch_length.append(len(t))
            batch_source.append(source)

    return {
        'input_ids': batch_input_ids,
        'source': batch_source,
        'length': batch_length
    }


def _concat_and_truncate(batch: dict[str, list[str | int]], max_length: int):    
    batch_input_ids = []
    batch_source = []
    batch_length = []

    current_source = batch['source'][0]
    current_input_ids = []
    for source, input_ids in zip(batch['source'], batch['input_ids']):
        if source != current_source:
            if current_input_ids:
                batch_input_ids.append(current_input_ids)
                batch_source.append(current_source)
                batch_length.append(len(current_input_ids))
                current_input_ids = []
            current_source = source

        current_input_ids += input_ids
        while len(current_input_ids) >= max_length:
            batch_input_ids.append(current_input_ids[:max_length])
            batch_source.append(current_source)
            batch_length.append(len(current_input_ids[:max_length]))
            current_input_ids[:max_length] = []
    
    if current_input_ids:
        batch_input_ids.append(current_input_ids)
        batch_source.append(current_source)
        batch_length.append(len(current_input_ids))

    return {
        'input_ids': batch_input_ids,
        'source': batch_source,
        'length': batch_length
    }
