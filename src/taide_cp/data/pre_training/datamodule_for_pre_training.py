from typing import cast

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from ..datamodule import DataModule
from .datacollator_for_pre_training import DataCollatorForPreTraining
from .pre_training_config import (ConcatMethod,
                                  PreTrainingConfig)


def _tokenize(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase):
    new_batch = tokenizer(
        [x for x in batch['text'] if x],
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False
    )

    for x in new_batch['input_ids']:        
        x.insert(0, tokenizer.bos_token_id)
        x.append(tokenizer.eos_token_id)

    return new_batch


def _split(batch: dict[str, list[list[int]]], max_length: int, stride: int):
    indices_to_split = []
    for i, x in enumerate(batch['input_ids']):
        if len(x) > max_length:
            indices_to_split.append(i)

    for i in sorted(indices_to_split, reverse=True):
        splited_input_ids_list = []
        original_input_ids = batch['input_ids'][i]
        for s in range(0, len(original_input_ids), stride):
            splited_input_ids = original_input_ids[s:s + max_length]
            splited_input_ids_list.append(splited_input_ids)
        
        batch['input_ids'][i:i + 1] = splited_input_ids_list

    batch['length'] = [len(x) for x in batch['input_ids']]
    return batch


def _concat_and_truncate(batch: dict[str, list[int]], max_length: int):
    new_batch = {'input_ids': []}
    
    input_ids = []
    for x in batch['input_ids']:
        input_ids += x
        while len(input_ids) >= max_length:
            new_batch['input_ids'] += [input_ids[:max_length]]
            input_ids[:max_length] = []

    if input_ids:
        new_batch['input_ids'] += [input_ids]
    
    return new_batch


def _group_by_length(lengths: list[int], max_length: int) -> list[list[int]]:
    groups = []
    current_group = []
    current_sum = 0
    
    for i, l in sorted(enumerate(lengths), key=lambda x: x[1]):
        if current_sum + l <= max_length:
            current_group.append(i)
            current_sum += l
        else:
            groups.append(current_group)
            current_group = [i]
            current_sum = l
    
    if current_group:
        groups.append(current_group)
    
    return groups


def _group_from_indices(batch: dict[str, list[list[int]]], dataset: Dataset):        
    return {
        'grouped_input_ids': [
            dataset[group]['input_ids'] for group in batch['grouped_indices']
        ]
    }


class DataModuleForPreTraining(DataModule):
    config: PreTrainingConfig

    def __init__(self, config: PreTrainingConfig) -> None:
        super().__init__(config)

        self.datacollator = DataCollatorForPreTraining(config)
 
    def _prepare_data(self, current_hook: str | None = None) -> DatasetDict:
        dataset_dict = super()._prepare_data(current_hook)

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns='text',
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
        )

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _split,
            batched=True,
            fn_kwargs=dict(
                max_length=self.config.max_length,
                stride=self.config.stride
            ),
            num_proc=self.config.num_proc
        )

        if self.config.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _concat_and_truncate,
                batched=True,
                remove_columns='length',
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc
            )
        elif self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
            for k, dataset in dataset_dict.items():
                dataset = cast(Dataset, dataset)
                group_dataset = Dataset.from_dict({'grouped_indices': _group_by_length(dataset['length'], self.config.max_length)})
                dataset_dict[k] = self.map_dataset(
                    group_dataset,
                    _group_from_indices,
                    batched=True,
                    remove_columns='grouped_indices',
                    fn_kwargs=dict(dataset=dataset),
                    num_proc=self.config.num_proc,
                    cache_file_name_fn=lambda x: dataset._get_cache_file_path(x)
                )
    
        return dataset_dict
