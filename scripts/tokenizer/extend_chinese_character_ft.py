import json
import os
import shutil
from pathlib import Path
from typing import List, Literal, Union

import fire

CCLevel = Literal['A', 'B', 'C', 'N']

def read_json(path: str, jsonl: bool = False) -> Union[dict, List[dict]]:
    with open(path, 'r', encoding='utf-8') as f:
        if jsonl:
            return [json.loads(l) for l in f]
        return json.load(f)
    
def write_json(path: str, obj: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=True)

def load_chinese_character(path: str, level: CCLevel) -> List[str]:
    cc = []
    for x in read_json(path, jsonl=True):
        if x['正字字形'] and x['正字字號'][0] <= level:
            cc.append(x['正字字形'])
    return cc

def main(
    tokenizer_path: str,
    cc_path: str,
    output_path: str,
    cc_level: CCLevel = 'B',
):
    tokenizer_path: Path = Path(tokenizer_path)
    output_path: Path = Path(output_path)
    tokenizer = read_json(tokenizer_path.joinpath('tokenizer.json'))
    
    cc = load_chinese_character(cc_path, cc_level)
    for c in cc:
        if c in tokenizer['model']['vocab']:
            continue
        tokenizer['model']['vocab'][c] = len(tokenizer['model']['vocab'])
    
    os.makedirs(output_path, exist_ok=True)
    write_json(output_path.joinpath('tokenizer.json'), tokenizer)
    exclusion = {'tokenizer.model', 'tokenizer.json'}
    for p in tokenizer_path.glob('*'):
        if p.name in exclusion:
            continue
        shutil.copy(p, output_path.joinpath(p.name))

if __name__ == '__main__':
    fire.Fire(main)
