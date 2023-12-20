from abc import ABC, abstractmethod
from functools import update_wrapper
from types import MethodType
from typing import Any, Callable, TypeVar
import functools


T = TypeVar('T')

class Patcher(ABC):
    def patch_method(self, method: MethodType, patched_method: Callable):
        target = method.__self__
        name = method.__name__
        assert method.__func__ is getattr(target.__class__, name), f'The method `{method.__qualname__}` seems to be patched already.'
        patched_method = update_wrapper(patched_method, method)
        patched_method = MethodType(patched_method, target)
        setattr(target, name, patched_method)

    def unpatch_method(self, method: MethodType):
        target = method.__self__
        name = method.__name__

        original_method = getattr(target.__class__, name)
        original_method = MethodType(original_method, target)
        if method.__func__ is not original_method:
            setattr(target, name, original_method)

    def patch_module(self, target: Any, module_path: str, module: Any):
        rsetattr(target, module_path, module)
    
    def _validate(self, target: T): ...

    @abstractmethod
    def patch(self, target: T): ...

    def unpatch(self):
        raise NotImplementedError()

    def __call__(self, target: T, *args, **kwargs):
        self._validate(target)
        self.patch(target, *args, **kwargs)
        return target

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class DummyPatcher(Patcher):
    def patch(self, target: T):
        return target


def rgetattr(obj, attr):
    return functools.reduce(getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
