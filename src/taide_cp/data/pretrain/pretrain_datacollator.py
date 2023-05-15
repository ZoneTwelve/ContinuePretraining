

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator

class PretrainDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.pad_token is not None

    def __call__(self, batch):
        batch = self.convert_list_of_dict_to_dict_of_list(batch)
        
        batch['input_ids'] = torch.tensor(batch['input_ids'])
        padding_mask = batch['input_ids'] == -1
        batch['input_ids'][padding_mask] = self.tokenizer.pad_token_id

        batch['attention_mask'] = torch.ones_like(padding_mask)
        batch['attention_mask'][padding_mask] = 0

        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][padding_mask] = -100
        return batch
