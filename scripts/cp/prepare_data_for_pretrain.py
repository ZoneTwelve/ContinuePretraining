import os
from glob import glob
from typing import Dict, List, Optional, Union

import fire
from datasets import (Dataset, concatenate_datasets, disable_caching,
                      load_dataset)
from transformers import AutoTokenizer, PreTrainedTokenizer

from taide_cp.utils import DatasetsContextManager


@DatasetsContextManager()
def load_datasets(data_path: str) -> Dataset:
    datasets = []
    paths = []
    if os.path.isfile(data_path):
        paths = [data_path]
    else:
        paths = glob(os.path.join(data_path, '**/*.*'), recursive=True)

    for p in paths:
        x = load_dataset('json', data_files=p)['train']
        datasets.append(x)

    return concatenate_datasets(datasets)

def tokenize(batch: List[str], tokenizer: PreTrainedTokenizer):
    batch = tokenizer(batch, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
    for input_ids in batch['input_ids']:
        input_ids[:] = [
            tokenizer.bos_token_id,
            *input_ids,
            tokenizer.eos_token_id
        ]
    return batch

def rearrange_datapoints(
    batch: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
    datapoints = []
    
    input_ids = []
    for x in batch['input_ids']:
        input_ids += x
        while len(input_ids) >= max_length:
            datapoint = input_ids[:max_length]
            datapoints.append(datapoint)
            input_ids[:max_length] = []

    if input_ids:
        paddings = [-1] * (max_length - len(input_ids))
        datapoint = paddings + input_ids if tokenizer.padding_side == 'left' else input_ids + paddings
        datapoints.append(datapoint)
    
    batch['input_ids'] = datapoints
    return batch

def main(
    data_path: str,
    tokenizer_path: str,
    output_path: str,
    max_length: int = 2048,
    num_proc: Optional[int] = None,
    test_size: Union[int, float] = 0.1,
):
    num_proc = num_proc or os.cpu_count()

    disable_caching()

    dataset = load_datasets(data_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, model_max_length=None)
    
    dataset = dataset.map(
        tokenize,
        fn_kwargs=dict(tokenizer=tokenizer),
        input_columns='text',
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=num_proc,
    )

    dataset = dataset.map(
        rearrange_datapoints,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        batched=True,
        num_proc=num_proc,
    )
    
    if test_size:
        dataset = dataset.train_test_split(test_size, seed=42)
    
    dataset.save_to_disk(output_path)

if __name__ == '__main__':
    fire.Fire(main)
