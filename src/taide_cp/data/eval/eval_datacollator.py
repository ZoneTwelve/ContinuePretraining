import torch
from transformers import PreTrainedTokenizerBase

from ..datacollator import DataCollator


class DataCollatorForEvaluation(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

    def __call__(self, batch):
        batch = self.convert_list_of_dict_to_dict_of_list(batch)
        
        batch_prompt_length = []
        batch_text = []
        for prompt, input, response in zip(batch['prompt'], batch['input'], batch['response']):
            input = '' if input in {None, '', '<noinput>'} else f'\n{input}'
            prompt = f'{self.tokenizer.bos_token}{prompt}{input}\n'
            prompt_length = len(self.tokenizer.tokenize(prompt))
            text = f'{prompt}{response}{self.tokenizer.eos_token}'

            batch_prompt_length.append(prompt_length)
            batch_text.append(text)

        batch_encoding = self.tokenizer(
            batch_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_token_type_ids=False,
            padding=True,
            # truncation=True,
            # max_length=2048
        )

        batch_encoding['labels'] = batch_encoding['input_ids'].clone()
        batch_padding_length = torch.count_nonzero(1 - batch_encoding['attention_mask'], dim=1)
        for i, labels in enumerate(batch_encoding['labels']):
            n = batch_padding_length[i] + batch_prompt_length[i]
            labels[:n] = -100

        return batch_encoding
