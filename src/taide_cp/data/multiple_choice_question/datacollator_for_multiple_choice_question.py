from typing import Any, Dict, List, Union

import opencc
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ..datacollator import DataCollator

t2s = opencc.OpenCC('t2s')

def padded_stack(tensors: list[torch.Tensor], value: int | float = 0) -> torch.Tensor:
    full_size = max([x.size(-1) for x in tensors])

    out = torch.stack(
        [
            F.pad(x, (0, full_size - x.size(-1)), value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    
    padding_mask = torch.zeros_like(out, dtype=torch.bool)
    for i, x in enumerate(tensors):
        padding_mask[i, x.size(-1):] = True
    
    return out, padding_mask


def pad_input_ids(input_ids: list[list[int]], pad_token_id: int):
    max_length = max(len(ii) for ii in input_ids)
    input_ids_ = []
    attention_mask = []
    for ii in input_ids:
        p = max_length - len(ii)
        attention_mask += [[1] * len(ii) + [0] * p]
        input_ids_ += [ii + [pad_token_id] * p]

    input_ids_ = torch.tensor(input_ids_)
    attention_mask = torch.tensor(attention_mask)
    return input_ids_, attention_mask


class DataCollatorForMultipleChoiceQuestion(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, convert_to_chs: bool = False) -> None:
        super().__init__(tokenizer)

        self.example = ''
        self.convert_to_chs = convert_to_chs

        self.pad_token_id = tokenizer.pad_token_id if 'pad_token' in tokenizer.special_tokens_map else tokenizer.eos_token_id
        assert self.pad_token_id is not None

    def get_prompt(
        self,
        x: Dict[str, Any],
        with_answer: Union[bool, str] = True,
        with_example: bool = True,
    ):
        s = ''
        s += self.example if with_example else ''
        s += f'題目：\n{x["question"]}\n答案：'
        if with_answer is True:
            s += x['choices'][x['answer']] + '\n'
        elif isinstance(with_answer, str):
            s += with_answer + '\n'
        return s
    
    def set_examples(self, examples: List[Dict[str, Any]]):
        self.example = ''
        for x in examples:
            self.example += self.get_prompt(x, with_example=False)
            self.example += '\n'

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        num_choices = len(batch[0]['choices'])
        
        batch_example = []
        batch_example_question = []
        batch_example_question_choices = [[] for _ in range(num_choices)]
        batch_choices = [[] for _ in range(num_choices)]
        batch_data = []

        # Some tokenizers don't add the BOS token (e.g., tokenizers of the MPT model),
        # but we always need one, so we add it manually.
        choice_context = f'{self.tokenizer.bos_token}'

        if self.convert_to_chs:
            choice_context = t2s.convert(choice_context)

        for x in batch:
            question = self.get_prompt(x, with_answer=False, with_example=False)
            example = self.example
            choices = x['choices']

            if self.convert_to_chs:
                question = t2s.convert(question)
                example = t2s.convert(self.example)
                choices = [t2s.convert(c) for c in choices]

            batch_example.append(example)
            batch_example_question.append(example + question)
            for i in range(num_choices):
                batch_example_question_choices[i].append(self.example + question + choices[i])
                batch_choices[i].append(choice_context + choices[i])

            batch_data.append({
                'id': x['id'],
                'example': self.example,
                'question': question,
                'choices': x['choices'],
                'answer': {'index': x['answer'], 'text': x['choices'][x['answer']]}
            })
        
        example_encoding = self.tokenizer(
            batch_example,
            return_tensors='pt',
            return_length=True,
            return_token_type_ids=False,
        )
        example_length = example_encoding.pop('length')[0]

        example_question_length = self.tokenizer(batch_example_question, return_length=True)['length']
        question_choice_inputs = []
        for batch_example_question_choice in batch_example_question_choices:
            # We want the padding side to always be right,
            # but some tokenizers don't support it (e.g., ChatGLM),
            # so we implement the padding logic ourselves.
            example_question_choice_encoding = self.tokenizer(batch_example_question_choice, return_token_type_ids=False)
            eqc_input_ids, eqc_attention_mask = pad_input_ids(example_question_choice_encoding['input_ids'], self.pad_token_id)
            
            qc_input_ids = eqc_input_ids[:, example_length:]
            padding_lengths = torch.count_nonzero(1 - eqc_attention_mask, dim=1)

            index = [torch.arange(eq_length - example_length, qc_input_ids.size(1) - p_length) for eq_length, p_length in zip(example_question_length, padding_lengths)]
            index, padding_mask = padded_stack(index)
            target = qc_input_ids.unsqueeze(-1).gather(1, index.unsqueeze(-1))
            index[~padding_mask] -= 1 # Shift for logits

            question_choice_inputs.append({
                'encoding': {
                    'input_ids': qc_input_ids,
                    'attention_mask': eqc_attention_mask
                },
                'choice_index': index,
                'choice_target': target,
                'choice_index_padding_mask': padding_mask,
            })

        choice_inputs = []
        choice_context_length = self.tokenizer([choice_context], return_length=True, add_special_tokens=False)['length'][0]
        for batch_choice in batch_choices:
            context_choice_encoding = self.tokenizer(batch_choice, return_token_type_ids=False, add_special_tokens=False)
            cc_input_ids, cc_attention_mask = pad_input_ids(context_choice_encoding['input_ids'], self.pad_token_id)
            padding_lengths = torch.count_nonzero(1 - cc_attention_mask, dim=1)
            
            index = [torch.arange(choice_context_length, cc_input_ids.size(1) - p) for p in padding_lengths]
            index, padding_mask = padded_stack(index)
            target = cc_input_ids.unsqueeze(-1).gather(1, index.unsqueeze(-1))
            index[~padding_mask] -= 1

            choice_inputs.append({
                'encoding': {
                    'input_ids': cc_input_ids,
                    'attention_mask': cc_attention_mask,
                },
                'target': target,
                'index': index,
                'index_padding_mask': padding_mask
            })

        return {
            'data': batch_data,
            'example_encoding': example_encoding if example_encoding['input_ids'].size(1) > 0 else None,
            'question_choice_inputs': question_choice_inputs,
            'choice_inputs': choice_inputs,
        }
