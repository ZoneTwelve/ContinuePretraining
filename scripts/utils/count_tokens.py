import os
from glob import glob
from typing import List, Optional

import fire
from datasets import (Dataset, concatenate_datasets, disable_caching,
                      load_dataset)
from transformers import PreTrainedTokenizer

from taide_cp.models import AutoTokenizer
from taide_cp.utils import DatasetsContextManager, cpu_count


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
    x = tokenizer(batch, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False, return_length=True)
    return x

def main(
    data_path: str,
    tokenizer_path: str,
    num_proc: Optional[int] = None,
):
    num_proc = num_proc or cpu_count()

    disable_caching()

    dataset = load_datasets(data_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=None)

    dataset = dataset.map(
        tokenize,
        fn_kwargs=dict(tokenizer=tokenizer),
        input_columns='text',
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=num_proc
    )

    print(sum(dataset['length']))

if __name__ == '__main__':
    fire.Fire(main)
