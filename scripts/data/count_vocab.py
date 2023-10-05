from datasets import load_from_disk
import torch
from transformers import AutoTokenizer
from collections import Counter

import multiprocess


def count(batch, vocab_size):
    counter = [0] * vocab_size
    for x in batch['input_ids']:
        for i in x:
            counter[i] += 1
    return {'counter': [counter]}


def main(
    tokenizer_path: str = 'checkpoints/tokenizer/llama-ccw',
    dataset_path: str = 'data/cp/tokenized/llama-ccw/k/tokenized',
    output_path: str = 'llama-ccw_k.pt',
    num_proc: int | None = 8
):
    multiprocess.set_start_method('spawn')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(
        count,
        fn_kwargs=dict(vocab_size=vocab_size),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=10000,
        num_proc=num_proc
    )

    counter = [0] * vocab_size
    for x in dataset:
        for i, c in enumerate(x['counter']):
            counter[i] += c

    counter = Counter({(i, tokenizer.convert_ids_to_tokens(i)): c for i, c in enumerate(counter)})    
    torch.save(counter, output_path)


if __name__ == '__main__':
    main()