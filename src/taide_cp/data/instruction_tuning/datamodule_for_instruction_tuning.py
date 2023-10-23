from string import Template
from typing import cast

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from ..datamodule import DataModule
from .datacollator_for_instruction_tuning import \
    DataCollatorForInstructionTuning
from .instruction_tuning_config import (ConcatMethod, InstructionTuningConfig,
                                        OverlongHandlingMethod)


def _apply_template(batch: dict[str, list[str]], prompt_template: Template, response_template: Template):
    batch: list[dict[str, str]] = [dict(zip(batch, x)) for x in zip(*batch.values())]

    new_batch = {
        'prompt': [],
        'response': [],
    }
    
    for x in batch:
        p = prompt_template.safe_substitute(**x)
        r = response_template.safe_substitute(**x)

        new_batch['prompt'] += [p]
        new_batch['response'] += [r]

    return new_batch

def _tokenize(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase):
    batch: list[dict[str, str]] = [dict(zip(batch, x)) for x in zip(*batch.values())]

    batch_prompt = []
    batch_prompt_response = []
    for x in batch:
        batch_prompt += [x['prompt']]
        batch_prompt_response += [x['prompt'] + x['response']]
    
    tokenizer_kwargs = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False
    )
    prompt_length = tokenizer(batch_prompt, return_length=True, **tokenizer_kwargs)['length']
    input_ids = tokenizer(batch_prompt_response, **tokenizer_kwargs)['input_ids']

    return {
        'input_ids': input_ids,
        'prompt_length': prompt_length
    }

def _drop_overlong(batch: dict[str, list], max_length: int):
    ids_to_drop = []
    for i, x in enumerate(batch['input_ids']):
        if len(x) > max_length:
            ids_to_drop.append(i)
    
    for i in sorted(ids_to_drop, reverse=True):
        del batch['input_ids'][i]
        del batch['prompt_length'][i]

def _truncate_overlong(batch: dict[str, list], max_length: int):
    for i, x in enumerate(batch['input_ids']):
        if len(x) > max_length:
            x[max_length:] = []

            if batch['prompt_length'][i] > max_length:
                batch['prompt_length'][i] = max_length

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
    new_batch = {
        'grouped_input_ids': [],
        'grouped_prompt_length': [],
    }
    for g in batch['grouped_indices']:
        x = dataset[g]
        new_batch['grouped_input_ids'] += [x['input_ids']]
        new_batch['grouped_prompt_length'] += [x['prompt_length']]
    return new_batch


class DataModuleForInstructionTuning(DataModule):
    config: InstructionTuningConfig

    def __init__(self, config: InstructionTuningConfig) -> None:
        super().__init__(config)

        self.datacollator = DataCollatorForInstructionTuning(config)
 
    def _prepare_data(self) -> DatasetDict:
        dataset_dict = super()._prepare_data()

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _apply_template,
            batched=True,
            remove_columns='text',
            fn_kwargs=dict(
                prompt_template=self.config.prompt_template,
                response_template=self.config.response_template
            ),
            num_proc=self.config.num_proc,
        )

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns='text',
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
        )

        if self.config.overlong_handling_method == OverlongHandlingMethod.DROP:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _drop_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
            )
        elif self.config.overlong_handling_method == OverlongHandlingMethod.TRUNCATE:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _truncate_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
            )
        
        if self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
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
