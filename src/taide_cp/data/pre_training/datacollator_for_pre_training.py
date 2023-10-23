import random
from typing import Any

import torch

from ..datacollator import DataCollator
from .pre_training_config import (ConcatMethod,
                                  PreTrainingConfig)


class DataCollatorForPreTraining(DataCollator):
    @property
    def tokenizer(self):
        return self.config.tokenizer

    def __init__(self, config: PreTrainingConfig) -> None:
        self.config = config

        assert config.tokenizer.pad_token is not None
    
    def _merge_grouped_input_ids(self, grouped_input_ids: list[list[int]]) -> list[int]:
        random.shuffle(grouped_input_ids)
        return [y for x in grouped_input_ids for y in x]

    def _pad_to_longest(self, x: list[list[int]]):
        n = max(len(y) for y in x)
        if self.config.pad_to_multiple_of is not None:
            n = ((n // self.config.pad_to_multiple_of) + 1) * self.config.pad_to_multiple_of
        
        for y in x:
            num_paddings = n - len(y)
            paddings = [-1] * num_paddings
            y[:] = paddings + y if self.tokenizer.padding_side == 'left' else y + paddings
        return x

    def __call__(self, batch: list[dict[str, Any]]):
        if self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
            input_ids = [self._merge_grouped_input_ids(x['grouped_input_ids']) for x in batch]
        else:
            input_ids = [x['input_ids'] for x in batch]
        
        input_ids = self._pad_to_longest(input_ids)
        input_ids = torch.tensor(input_ids)
        padding_mask = input_ids == -1
        input_ids[padding_mask] = self.tokenizer.pad_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids).masked_fill(padding_mask, 0),
            'labels': input_ids.masked_fill(padding_mask, -100)
        }
