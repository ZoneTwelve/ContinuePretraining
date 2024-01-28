import random
from typing import Any

import torch

from ..datacollator import DataCollator
from .datamodule_for_instruction_tuning_config import (ConcatMethod,
                                                       DataModuleForInstructionTuningConfig)


class DataCollatorForInstructionTuning(DataCollator):
    config: DataModuleForInstructionTuningConfig
    
    @property
    def tokenizer(self):
        return self.config.tokenizer

    def __init__(self, config: DataModuleForInstructionTuningConfig):
        super().__init__(config)
    
    def _merge_grouped(self, x: list[list[int]], indices: list[int]) -> list[int]:
        return [y for i in indices for y in x[i]]

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
        batch_input_ids = []
        batch_labels = []
        
        for x in batch:
            if self.config.concat_method == ConcatMethod.NO_CONCAT:
                input_ids = x['input_ids']
                labels = x['labels']
            elif self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
                indices = list(range(len(x['grouped_input_ids'])))
                random.shuffle(indices)
                input_ids = self._merge_grouped(x['grouped_input_ids'], indices)
                labels = self._merge_grouped(x['grouped_labels'], indices)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        batch_input_ids = self._pad_to_longest(batch_input_ids)
        batch_labels = self._pad_to_longest(batch_labels)

        batch_input_ids = torch.tensor(batch_input_ids)
        padding_mask = batch_input_ids == -1
        batch_input_ids[padding_mask] = self.tokenizer.pad_token_id

        batch_labels = torch.tensor(batch_labels).masked_fill(padding_mask, -100)
        attention_mask = torch.ones_like(batch_input_ids).masked_fill(padding_mask, 0)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': attention_mask,
            'labels': batch_labels
        }
