import copy
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ..datacollator import DataCollator

ANSWER_KEY_MAPPING = {
    1: '選項一',
    2: '選項二',
    3: '選項三',
    4: '選項四',
}


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
        # s += f'題目：\n{x["題目"]}\n選項：\n1. {x["選項一"]}\n2. {x["選項二"]}\n3. {x["選項三"]}\n4. {x["選項四"]}\n答案：'
        s += f'題目：\n{x["題目"]}\n答案：'
        if with_answer is True:
            s += x[ANSWER_KEY_MAPPING[x["正確答案"]]] + '\n'
        elif isinstance(with_answer, str):
            s += with_answer + '\n'
        return s
    
    def set_examples(self, examples: List[Dict[str, Any]]):
        self.example = ''
        for x in examples:
            self.example += self.get_prompt(x, with_example=False)
            self.example += '\n'

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        batch_example = []
        batch_example_question = []
        batch_example_question_choices = [[], [], [], []]
        
        batch_id = []
        batch_question = []
        batch_choices = []
        batch_answer = []

        for x in batch:
            batch_id.append(x['id'])
            batch_example.append(self.example)
            
            q = self.get_prompt(x, with_answer=False, with_example=False)

            batch_example_question.append(self.example + q)

            batch_example_question_choices[0].append(self.example + q + x['選項一'])
            batch_example_question_choices[1].append(self.example + q + x['選項二'])
            batch_example_question_choices[2].append(self.example + q + x['選項三'])
            batch_example_question_choices[3].append(self.example + q + x['選項四'])

            batch_question.append(x['題目'])

            batch_choices.append([
                x['選項一'],
                x['選項二'],
                x['選項三'],
                x['選項四']
            ])

            batch_answer.append({
                'index': x['正確答案'] - 1,
                'text': x[ANSWER_KEY_MAPPING[x['正確答案']]]
            })

        tokenizer_kwargs = dict(
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True
        )
        
        example_encoding = self.tokenizer(batch_example, return_length=True, **tokenizer_kwargs)
        example_length = example_encoding.pop('length')[0]

        example_question_choice_encodings = []
        for batch_example_question_choice in batch_example_question_choices:
            example_question_choice_encoding = self.tokenizer(batch_example_question_choice, **tokenizer_kwargs)
            example_question_choice_encodings.append(example_question_choice_encoding)

        
        question_choice_encodings = []
        for example_question_choice_encoding in example_question_choice_encodings:
            input_ids = example_question_choice_encoding['input_ids'][:, example_length:]            
            attention_mask = example_question_choice_encoding['attention_mask']

            question_choice_encodings.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })

        example_question_encoding = self.tokenizer(batch_example_question, return_length=True)
        
        choice_targets = []
        choice_indices = []
        choice_index_padding_masks = []
        for question_choice_encoding in question_choice_encodings:
            input_ids = question_choice_encoding['input_ids']
            attention_mask = question_choice_encoding['attention_mask'][:, example_length:]
            num_paddings = torch.count_nonzero(1 - attention_mask, dim=1)

            index = []
            for i in range(input_ids.size(0)):
                s = example_question_encoding['length'][i] - example_length
                e = input_ids.size(1) - num_paddings[i].item()
                index.append(torch.arange(s, e))
        
            index, padding_mask = padded_stack(index)
            target = input_ids.unsqueeze(-1).gather(1, index.unsqueeze(-1))
            index[~padding_mask] -= 1 # Shift for logits

            choice_targets.append(target)
            choice_indices.append(index)
            choice_index_padding_masks.append(padding_mask)

        return {
            'id': batch_id,
            'example': batch_example,
            'question': batch_question,
            'choices': batch_choices,
            'answer': batch_answer,
            'example_encoding': example_encoding,
            'question_choice_encodings': question_choice_encodings,
            'choice_targets': choice_targets,
            'choice_indices': choice_indices,
            'choice_index_padding_masks': choice_index_padding_masks,
        }
