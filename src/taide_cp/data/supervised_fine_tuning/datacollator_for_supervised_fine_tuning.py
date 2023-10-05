from typing import Any

import torch
from transformers import PreTrainedTokenizer

from ..datacollator import DataCollator
from .templater import PromptTemplater, Templater


class DataCollatorForSupervisedFineTuning(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, templaters: list[Templater] = [], max_length: int | None = None) -> None:
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.prompt_templaters: list[PromptTemplater] = []
        self.templaters: list[Templater] = []

        for templater in templaters:
            if isinstance(templater, PromptTemplater):
                self.prompt_templaters += [templater]
            else:
                self.templaters += [templater]

    def __call__(self, batch: list[dict[str, Any]]):        
        batch_prompt_length = []
        batch_text = []
    
        for x in batch:
            for templater in self.templaters:
                if templater.match(**x):
                    prompt, response = templater.apply(**x)
                    break
            
            for templater in self.prompt_templaters:
                if templater.match(**x):
                    prompt, response = templater.apply(
                        prompt=prompt,
                        response=response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token
                    )
            prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            
            text = prompt + response
            batch_prompt_length += [prompt_length]
            batch_text += [text]

        batch_encoding = self.tokenizer(
            batch_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        batch_encoding['labels'] = batch_encoding['input_ids'].clone()
        batch_padding_length = torch.count_nonzero(1 - batch_encoding['attention_mask'], dim=1)
        for i in range(len(batch)):
            if self.tokenizer.padding_side == 'left':
                n = batch_padding_length[i] + batch_prompt_length[i]
                batch_encoding['labels'][i, :n] = -100
            elif self.tokenizer.padding_side == 'right':
                batch_encoding['labels'][i, :batch_prompt_length[i]] = -100
                batch_encoding['labels'][i, :-batch_padding_length[i]] = -100
        
        return batch_encoding
