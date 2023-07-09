import functools
import json
import logging
import os
from contextlib import AbstractContextManager, ContextDecorator, ExitStack, contextmanager
import sys
from typing import (Any, Callable, ContextManager, Dict, List, ParamSpec, Type,
                    TypeVar, Union)

__all__ = [
    'read_json',
    'write_json',
    'rgetattr',
    'rsetattr',
    'DatasetsContextManager',
    'ContextManagers',
    'parse_ev',
    'copy_callable_signature',
    'cpu_count',
    'disable_output',
]

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

class DatasetsContextManager(ContextDecorator):
    @staticmethod
    def set_caching_enabled(boolean: bool):
        import datasets

        if boolean:
            datasets.enable_caching()
        else:
            datasets.disable_caching()

    @staticmethod
    def set_progress_bar_enabled(boolean: bool):
        import datasets

        if boolean:
            datasets.enable_progress_bar()
        else:
            datasets.disable_progress_bar()

    @classmethod
    def get_state(cls):
        import datasets
        return {
            'caching': datasets.is_caching_enabled(),
            'progress_bar': datasets.is_progress_bar_enabled(),
            'verbosity': datasets.logging.get_verbosity(),
        }

    @classmethod
    def set_state(cls, state: Dict[str, Any]):
        import datasets
        cls.set_caching_enabled(state['caching'])
        cls.set_progress_bar_enabled(state['progress_bar'])
        datasets.logging.set_verbosity(state['verbosity'])

    def __init__(
        self,
        caching: bool = False,
        progress_bar: bool = False,
        verbosity: int = logging.ERROR,
    ) -> None:
        super().__init__()

        self.old_state = None
        self.new_state = {
            'caching': caching,
            'progress_bar': progress_bar,
            'verbosity': verbosity,
        }
    
    def __enter__(self):
        self.old_state = self.get_state()
        self.set_state(self.new_state)
    
    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.set_state(self.old_state)

def parse_ev(type_: Type[T1], ev_name: str, default: T2 = None) -> Union[T1, T2]:
    ev = os.environ.get(ev_name)
    return type_(ev) if ev is not None else default


def copy_callable_signature(
    source: Callable[P, T1]
) -> Callable[[Callable[..., T1]], Callable[P, T1]]:
    def wrapper(target: Callable[..., T1]) -> Callable[P, T1]:
        @functools.wraps(source)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T1:
            return target(*args, **kwargs)
        return wrapped
    return wrapper

def cpu_count() -> int:
    try:
        import subprocess
        return int(subprocess.run('nproc', stdout=subprocess.PIPE).stdout)
    except FileNotFoundError:
        import psutil
        return psutil.cpu_count(logical=False)

@contextmanager
def disable_output():
    sys.stdout = open(os.devnull, 'w')
    yield
    sys.stdout.close()
    sys.stdout = sys.__stdout__
