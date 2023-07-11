import copy
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ..datacollator import DataCollator


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


class DataCollatorForMultipleChoiceQuestion(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        tokenizer = copy.copy(tokenizer)
        super().__init__(tokenizer)

        self.tokenizer.padding_side = 'right'
        self.example = ''

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
        batch_answers = [[] for _ in range(num_choices)]
        batch_data = []

        answer_context = '答案：'

        for x in batch:
            q = self.get_prompt(x, with_answer=False, with_example=False)
            
            batch_example.append(self.example)

            batch_example_question.append(self.example + q)

            for i in range(num_choices):
                batch_example_question_choices[i].append(self.example + q + x['choices'][i])
                batch_answers[i].append(answer_context + x['choices'][i])

            batch_data.append({
                'id': x['id'],
                'example': self.example,
                'question': q,
                'choices': x['choices'],
                'answer': {'index': x['answer'], 'text': x['choices'][x['answer']]}
            })

        tokenizer_kwargs = dict(
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True
        )
        
        example_encoding = self.tokenizer(batch_example, return_length=True, **tokenizer_kwargs)
        example_length = example_encoding.pop('length')[0]

        example_question_length = self.tokenizer(batch_example_question, return_length=True)['length']
        question_choice_inputs = []
        for batch_example_question_choice in batch_example_question_choices:
            example_question_choice_encoding = self.tokenizer(batch_example_question_choice, **tokenizer_kwargs)
            question_choice_encoding = {
                'input_ids': example_question_choice_encoding['input_ids'][:, example_length:].clone(),
                'attention_mask': example_question_choice_encoding['attention_mask'].clone(),
            }
            num_paddings = torch.count_nonzero(1 - example_question_choice_encoding['attention_mask'], dim=1)

            qc_input_ids = question_choice_encoding['input_ids']
            index = [torch.arange(eq_l - example_length, qc_input_ids.size(1) - p) for eq_l, p in zip(example_question_length, num_paddings)]
            index, padding_mask = padded_stack(index)
            target = qc_input_ids.unsqueeze(-1).gather(1, index.unsqueeze(-1))
            index[~padding_mask] -= 1 # Shift for logits

            question_choice_inputs.append({
                'encoding': question_choice_encoding,
                'choice_index': index,
                'choice_target': target,
                'choice_index_padding_mask': padding_mask,
            })

        answer_inputs = []
        answer_context_length = self.tokenizer(answer_context, return_length=True)['length']
        for batch_answer in batch_answers:
            encoding = self.tokenizer(batch_answer, **tokenizer_kwargs)
            input_ids = encoding['input_ids']
            num_paddings = torch.count_nonzero(1 - encoding['attention_mask'], dim=1)
            
            index = [torch.arange(answer_context_length, input_ids.size(1) - p) for p in num_paddings]
            index, padding_mask = padded_stack(index)
            target = input_ids.unsqueeze(-1).gather(1, index.unsqueeze(-1))
            index[~padding_mask] -= 1

            answer_inputs.append({
                'encoding': encoding,
                'target': target,
                'index': index,
                'index_padding_mask': padding_mask
            })

        return {
            'data': batch_data,
            'example_encoding': example_encoding,
            'question_choice_inputs': question_choice_inputs,
            'answer_inputs': answer_inputs,
        }
