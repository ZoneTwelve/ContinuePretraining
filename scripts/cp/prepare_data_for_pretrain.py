import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
from datasets import (Dataset, concatenate_datasets, disable_caching,
                      load_dataset)
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_datasets(data_dir: Union[str, Path]) -> Dataset:
    data_dir = Path(data_dir)
    datasets = []
    for p in data_dir.glob('*'):
        x = load_dataset('json', data_files=str(p))['train']
        datasets.append(x)
    return concatenate_datasets(datasets)

def tokenize(batch: List[str], tokenizer: PreTrainedTokenizerFast):
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
    tokenizer: PreTrainedTokenizerFast,
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
    data_dir: str,
    tokenizer_path: str,
    max_length: int,
    output_path: str,
    num_proc: Optional[int] = 4,
    test_size: Union[int, float] = 10000,
):
    num_proc = num_proc or os.cpu_count()

    disable_caching()

    dataset = load_datasets(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    dataset = dataset.map(
        tokenize,
        fn_kwargs=dict(tokenizer=tokenizer),
        input_columns='text',
        remove_columns=dataset.column_names,
        batched=True,
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