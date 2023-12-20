import logging

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from ..datamodule import DataModule
from .datacollator_for_pre_training import DataCollatorForPreTraining
from .datamodule_for_pre_training_config import ConcatMethod, DataModuleForPreTrainingConfig

logger = logging.getLogger(__name__)


class DataModuleForPreTraining(DataModule):
    config: DataModuleForPreTrainingConfig

    def __init__(self, config: DataModuleForPreTrainingConfig) -> None:
        super().__init__(config)

        self.datacollator = DataCollatorForPreTraining(config)

    def _tokenize(self, dataset_dict: DatasetDict):
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

        logger.debug(f'[rank: {global_rank}] Tokenize')
        dataset_dict = self._tokenize(dataset_dict)

        logger.debug(f'[rank: {global_rank}] Concat')
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
    
        return dataset_dict
    
    def count_tokens(self) -> dict[str, int]:
        dataset_dict = super().load_data()
        dataset_dict = self._tokenize(dataset_dict)
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
 

def _count_tokens(batch: dict[str, list[list[int]]]):
    return {
        'tokens': [sum(len(x) for x in batch['input_ids'])]
    }
