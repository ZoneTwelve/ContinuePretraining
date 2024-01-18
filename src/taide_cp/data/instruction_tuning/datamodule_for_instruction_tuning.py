from string import Template

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from ..datamodule import DataModule
from .datacollator_for_instruction_tuning import \
    DataCollatorForInstructionTuning
from .instruction_tuning_config import (ConcatMethod, InstructionTuningConfig,
                                        OverlongHandlingMethod)


class DataModuleForInstructionTuning(DataModule):
    config: InstructionTuningConfig
    datacollator_class = DataCollatorForInstructionTuning

    def __init__(self, config: InstructionTuningConfig) -> None:
        super().__init__(config)
 
    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _apply_template,
            batched=True,
            remove_columns=True,
            fn_kwargs=dict(
                prompt_template=self.config.prompt_template,
                response_template=self.config.response_template
            ),
            num_proc=self.config.num_proc,
            desc='Apply template'
        )

        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns=['prompt', 'response'],
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
            desc='Tokenize'
        )

        if self.config.overlong_handling_method == OverlongHandlingMethod.DROP:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _drop_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Drop overlong'
            )
        elif self.config.overlong_handling_method == OverlongHandlingMethod.TRUNCATE:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _truncate_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Truncate overlong'
            )
        
        if self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _group_by_length,
                batched=True,
                batch_size=10000,
                remove_columns=['input_ids', 'prompt_length'],
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Group by length'
            )
    
        return dataset_dict


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
    return batch

def _truncate_overlong(batch: dict[str, list], max_length: int):
    for i, x in enumerate(batch['input_ids']):
        if len(x) > max_length:
            x[max_length:] = []

            if batch['prompt_length'][i] > max_length:
                batch['prompt_length'][i] = max_length
    return batch

def _group_indices_by_length(lengths: list[int], max_length: int) -> list[list[int]]:
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

def _group_by_length(batch: dict[str, list[list[int]]], max_length: int):
    new_batch = {
        'grouped_input_ids': [],
        'grouped_prompt_length': [],
    }

    groups = _group_indices_by_length([len(x) for x in batch['input_ids']], max_length)
    for group in groups:
        grouped_input_ids = []
        grouped_prompt_length = []
        for i in group:
            grouped_input_ids.append(batch['input_ids'][i])
            grouped_prompt_length.append(batch['prompt_length'][i])
        new_batch['grouped_input_ids'].append(grouped_input_ids)
        new_batch['grouped_prompt_length'].append(grouped_prompt_length)
    return new_batch
