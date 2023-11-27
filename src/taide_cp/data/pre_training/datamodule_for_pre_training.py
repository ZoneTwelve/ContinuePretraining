import logging

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from ..datamodule import DataModule
from .datacollator_for_pre_training import DataCollatorForPreTraining
from .pre_training_config import ConcatMethod, PreTrainingConfig

logger = logging.getLogger(__name__)

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


def _split_overlong(batch: dict[str, list[list[int]]], max_length: int, stride: int):
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
    return batch

def _drop_overlong(batch: dict[str, list[list[int]]], max_length: int):
    indices_to_drop = []
    for i, x in enumerate(batch['input_ids']):
        if len(x) > max_length:
            indices_to_drop.append(i)

    for i in sorted(indices_to_drop, reverse=True):
        del batch['input_ids'][i]
    
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
    groups = _group_indices_by_length([len(x) for x in batch['input_ids']], max_length)
    return {
        'grouped_input_ids': [
            [batch['input_ids'][i] for i in group] for group in groups
        ]
    }

def _count_tokens(batch: dict[str, list[list[int]]]):
    return {
        'tokens': [sum(len(x) for x in batch['input_ids'])]
    }

class DataModuleForPreTraining(DataModule):
    config: PreTrainingConfig

    def __init__(self, config: PreTrainingConfig) -> None:
        super().__init__(config)

        self.datacollator = DataCollatorForPreTraining(config)

    def _process_data_tokenize(self, dataset_dict: DatasetDict):
        return self.map_dataset_dict(
            dataset_dict,
            _tokenize,
            batched=True,
            remove_columns=True,
            fn_kwargs=dict(tokenizer=self.config.tokenizer),
            num_proc=self.config.num_proc,
            desc='Tokenize'
        )

    def process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        global_rank = self.trainer.global_rank if self.trainer else 0

        logger.debug(f'[rank: {global_rank}] Tokenizing')
        dataset_dict = self._process_data_tokenize(dataset_dict)
        logger.debug(f'[rank: {global_rank}] Done.')
        
        logger.debug(f'[rank: {global_rank}] Handling overlong')
        if self.config.drop_overlong:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _drop_overlong,
                batched=True,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Drop overlong'
            )
        else:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _split_overlong,
                batched=True,
                fn_kwargs=dict(
                    max_length=self.config.max_length,
                    stride=self.config.stride
                ),
                num_proc=self.config.num_proc,
                desc='Split overlong'
            )
        logger.debug(f'[rank: {global_rank}] Done.')

        logger.debug(f'[rank: {global_rank}] Concating')
        if self.config.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _concat_and_truncate,
                batched=True,
                batch_size=100000,
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Concat and truncate'
            )
        elif self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
            dataset_dict = self.map_dataset_dict(
                dataset_dict,
                _group_by_length,
                batched=True,
                batch_size=10000,
                remove_columns='input_ids',
                fn_kwargs=dict(max_length=self.config.max_length),
                num_proc=self.config.num_proc,
                desc='Group by length'
            )
        logger.debug(f'[rank: {global_rank}] Done.')
    
        return dataset_dict
    
    def count_tokens(self) -> dict[str, int]:
        dataset_dict = super().load_data()
        dataset_dict = self._process_data_tokenize(dataset_dict)
        dataset_dict = self.map_dataset_dict(
            dataset_dict,
            _count_tokens,
            batched=True,
            batch_size=100000,
            remove_columns='input_ids',
            num_proc=self.config.num_proc,
            desc='Count tokens'
        )
        return {k: sum(x['tokens'] for x in v) for k, v in dataset_dict.items()}
