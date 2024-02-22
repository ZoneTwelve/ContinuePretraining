from abc import ABC, abstractmethod
from functools import update_wrapper, partial
from types import MethodType
from typing import Any, Callable, TypeVar
import functools


T = TypeVar('T')

class Patcher(ABC):
    def patch_method(self, method: MethodType | partial, patched_method: Callable) -> None:
        if isinstance(method, partial) and method.__name__ == 'forward' and hasattr(method.args[0], '_old_forward'):
            name = '_old_forward'
            target = method.args[0]
            original_func = target._old_forward.__func__
        else:
            name = method.__name__
            target = method.__self__
            original_func = method.__func__

        assert original_func is getattr(target.__class__, method.__name__), f'The method `{method.__qualname__}` seems to be patched already.'
        patched_method = update_wrapper(patched_method, method)
        patched_method = MethodType(patched_method, target)
        setattr(target, name, patched_method)

    def unpatch_method(self, method: MethodType) -> None:
        target = method.__self__
        name = method.__name__

        original_method = getattr(target.__class__, name)
        original_method = MethodType(original_method, target)
        if method.__func__ is not original_method:
            setattr(target, name, original_method)

    def patch_module(self, target: Any, module_path: str, module: Any) -> None:
        rsetattr(target, module_path, module)
    
    def _validate(self, target: T) -> None: ...

    @abstractmethod
    def patch(self, target: T) -> T: ...

    def unpatch(self, target: T) -> T:
        raise NotImplementedError()

    def __call__(self, target: T, *args, **kwargs) -> T:
        self._validate(target)
        self.patch(target, *args, **kwargs)
        return target

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        extra_repr = self.extra_repr()
        lines = extra_repr.split('\n') if extra_repr else None
        main_str = self.__class__.__name__ + '('
        if lines:
            if len(lines) == 1:
                main_str += lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class DummyPatcher(Patcher):
    def patch(self, target: T) -> T:
        return target


def rgetattr(obj, attr):
    return functools.reduce(getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
