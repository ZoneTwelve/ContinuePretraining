import glob
import os
from typing import Dict, Optional, Tuple, Union

import fire
from datasets import Dataset, concatenate_datasets, load_dataset
from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers
from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tqdm.auto import tqdm


class CustomSentencePieceBPETokenizer(SentencePieceBPETokenizer):
    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        cache_capacity: Optional[int] = None,
        unk_token: Union[str, AddedToken] = '<unk>',
        replacement: str = '▁',
        add_prefix_space: bool = True,
        dropout: Optional[float] = None,
        fuse_unk: Optional[bool] = False,
        byte_fallback: bool = False
    ):
        if vocab is not None and merges is not None:
            tokenizer = Tokenizer(BPE(
                vocab,
                merges,
                cache_capacity=cache_capacity,
                dropout=dropout,
                unk_token=unk_token,
                fuse_unk=fuse_unk,
                byte_fallback=byte_fallback
            ))
        else:
            tokenizer = Tokenizer(BPE(
                cache_capacity=cache_capacity,
                dropout=dropout,
                unk_token=unk_token,
                fuse_unk=fuse_unk,
                byte_fallback=byte_fallback
            ))

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(individual_digits=True),
        ])
        tokenizer.decoder = decoders.Sequence([
            decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            decoders.ByteFallback(),
        ])

        parameters = {
            'model': 'SentencePieceBPE',
            'unk_token': unk_token,
            'replacement': replacement,
            'add_prefix_space': add_prefix_space,
            'dropout': dropout,
        }

        super(SentencePieceBPETokenizer, self).__init__(tokenizer, parameters)


def load_datasets(data_path: str) -> Dataset:
    datasets = []
    paths = []
    if os.path.isdir(data_path):
        paths = glob.glob(os.path.join(data_path, '**/*.*'), recursive=True)
        paths = list(filter(lambda p: os.path.isfile(p), paths))
    else:
        paths = [data_path]

    progress = tqdm(total=len(paths), desc='Loading Files', leave=False)
    for p in paths:
        if os.path.isfile(p):
            x = load_dataset('json', data_files=p)['train']
        else:
            x = load_dataset(p)['train']
        datasets.append(x)
        progress.update()

    return concatenate_datasets(datasets)


def main(
    data_path: str,
    output_path: str,
    cache_capacity: int | None = None,
    vocab_size: int = 30000,
    min_frequency: int = 2,
):
    dataset = load_datasets(data_path)
    dataset = dataset.shuffle()

    tokenizer = CustomSentencePieceBPETokenizer(
        cache_capacity=cache_capacity,
        byte_fallback=True
    )
    tokenizer.train_from_iterator(
        (x['text'] for x in dataset),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        length=dataset.num_rows
    )
    tokenizer.save(output_path)


if __name__ == '__main__':
    fire.Fire(main)
