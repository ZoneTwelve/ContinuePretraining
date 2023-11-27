import random
from typing import Any

import torch

from ..datacollator import DataCollator
from .instruction_tuning_config import ConcatMethod, InstructionTuningConfig


class DataCollatorForInstructionTuning(DataCollator):
    @property
    def tokenizer(self):
        return self.config.tokenizer

    def __init__(self, config: InstructionTuningConfig):
        super().__init__()

        self.config = config

    def _merge_grouped(self, x: dict[str, list]) -> tuple[list[int], list[int]]:
        x = list(zip(x['grouped_input_ids'], x['grouped_prompt_length']))
        random.shuffle(x)

        merged_input_ids = []
        merged_labels = []
        for input_ids, prompt_length in x:
            merged_input_ids += input_ids
            labels = input_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            merged_labels += labels

        return merged_input_ids, merged_labels

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
        input_ids = []
        labels = []
        for x in batch:
            if self.config.concat_method == ConcatMethod.NO_CONCAT:
                pl = x['prompt_length']
                x = x['input_ids']
                y = x.copy()
                y[:pl] = [-100] * pl
            elif self.config.concat_method == ConcatMethod.GROUP_BY_LENGTH:
                x, y = self._merge_grouped(x)

            input_ids += [x]
            labels += [y]

        input_ids = self._pad_to_longest(input_ids)
        labels = self._pad_to_longest(labels)

        input_ids = torch.tensor(input_ids)
        padding_mask = input_ids == -1
        input_ids[padding_mask] = self.tokenizer.pad_token_id

        labels = torch.tensor(labels).masked_fill(padding_mask, -100)
        attention_mask = torch.ones_like(input_ids).masked_fill(padding_mask, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
