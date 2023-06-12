import functools
import json
from typing import Any

__all__ = ['read_json', 'write_json', 'rgetattr', 'rsetattr']

def read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=True)

def rgetattr(obj: Any, attr: str):
    return functools.reduce(getattr, [obj] + attr.split('.'))

def rsetattr(obj: Any, attr: str, val: Any):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
