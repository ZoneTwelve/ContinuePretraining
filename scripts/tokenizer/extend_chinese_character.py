import json
import os
import shutil
from pathlib import Path
from typing import List, Literal, Union

from transformers import AutoTokenizer

import fire

CCLevel = Literal['A', 'B', 'C', 'N']

def read_json(path: str, jsonl: bool = False) -> Union[dict, List[dict]]:
    with open(path, 'r', encoding='utf-8') as f:
        if jsonl:
            return [json.loads(l) for l in f]
        return json.load(f)

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    cc = load_chinese_character(cc_path, cc_level)
    tokenizer.add_tokens(cc)
    tokenizer.save_pretrained(output_path)
        

if __name__ == '__main__':
    fire.Fire(main)
