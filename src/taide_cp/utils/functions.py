import functools
import json
import logging
from contextlib import ContextDecorator
from typing import Any, Dict

__all__ = ['read_json', 'write_json', 'rgetattr', 'rsetattr', 'DatasetsContextManager']

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

        import datasets
        
        self.old_state = {
            'caching': datasets.is_caching_enabled(),
            'progress_bar': datasets.is_progress_bar_enabled(),
            'verbosity': datasets.logging.get_verbosity(),
        }

        self.new_state = {
            'caching': caching,
            'progress_bar': progress_bar,
            'verbosity': verbosity,
        }
    
    def __enter__(self):
        self.set_state(self.new_state)
    
    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.set_state(self.old_state)
