import json
from pathlib import Path
import shutil
from typing import Union

import fire


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def write_json(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=True)

def main(
    tokenizer_path: Union[str, Path],
    new_tokenizer_file: str,
    output_path: Union[str, Path],
):
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path)

    t1 = read_json(tokenizer_path.joinpath('tokenizer.json'))
    t2 = read_json(new_tokenizer_file)
    
    vocab: dict = t1['model']['vocab']
    merges: list = t1['model']['merges']
    merges_set = set(merges)

    for v in t2['model']['vocab']:
        if v not in vocab:
            vocab[v] = len(vocab)

    for m in t2['model']['merges']:
        if m not in merges_set:
            merges.append(m)
    
    output_path.mkdir(exist_ok=True)
    write_json(output_path.joinpath('tokenizer.json'), t1)
    for p in tokenizer_path.iterdir():
        if p.name != 'tokenizer.json':
            shutil.copy(p, output_path.joinpath(p.name))

if __name__ == '__main__':
    fire.Fire(main)
