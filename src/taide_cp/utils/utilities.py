import functools
import json
import os
from contextlib import AbstractContextManager, ExitStack
from typing import Any, ContextManager, List, ParamSpec, Type, TypeVar, Union

T1 = TypeVar('T1')
T2 = TypeVar('T2')
P = ParamSpec('P')

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

class ContextManagers(AbstractContextManager):
    def __init__(self, context_managers: List[ContextManager]) -> None:
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self
    
    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.stack.close()

def parse_ev(type_: Type[T1], ev_name: str, default: T2 = None) -> Union[T1, T2]:
    ev = os.environ.get(ev_name)
    return type_(ev) if ev is not None else default

def cpu_count() -> int:
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))

    cpu_count = os.cpu_count()
    return 1 if cpu_count is None else cpu_count
