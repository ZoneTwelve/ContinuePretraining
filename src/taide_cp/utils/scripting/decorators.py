import logging
from functools import wraps
from inspect import Parameter, _ParameterKind, signature
from typing import Callable, Iterable, Optional, Union, TypeVar


__all__ = ['component', 'use', 'entry_point']

_T = TypeVar('_T')

IGNORED_KEYWORDS_ON_ENTRY_POINT_ATTR_NAME = '__ignored_keywords_on_entry_point__'

def has_kind(parameters: Iterable[Parameter], kinds: Union[_ParameterKind, Iterable[_ParameterKind]]):
    kinds = [kinds] if not isinstance(kinds, Iterable) else kinds
    kinds = set(kinds)
    for p in parameters:
        if p.kind in kinds:
            return True
    return False

def component(ignored_keywords_on_entry_point: Optional[Iterable[str]] = None):
    def decorator(func: _T) -> _T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            parameters = signature(func).parameters
            if not has_kind(parameters.values(), Parameter.VAR_KEYWORD):
                kwargs = {k: v for k, v in kwargs.items() if k in parameters}
            return func(*args, **kwargs)
        
        if ignored_keywords_on_entry_point is not None:
            setattr(wrapper, IGNORED_KEYWORDS_ON_ENTRY_POINT_ATTR_NAME, set(ignored_keywords_on_entry_point))
        
        return wrapper
    return decorator

def use(component: Callable, ignored: Optional[Iterable[str]] = None):
    ignored = set(ignored) if ignored is not None else set()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        p1 = [p for p in signature(func).parameters.values()]
        p2 = [p for p in signature(component).parameters.values() if p.kind != Parameter.VAR_KEYWORD]

        assert not has_kind(p2, [Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL]), 'Functions with positional-only arguments are unsupported'

        k1 = {p.name for p in p1 if p.kind != Parameter.VAR_KEYWORD}
        k2 = {p.name for p in p2 if p.kind != Parameter.VAR_KEYWORD}
        same_kw_names = k1 & k2
        if same_kw_names:
            logging.debug('\n'.join(f'{func.__name__}.{k} and {component.__name__}.{k} has same keyword name' for k in same_kw_names))

        p1 = [p for p in p1 if p.kind != Parameter.VAR_KEYWORD]
        p2 = [p for p in p2 if p.name not in same_kw_names and p.name not in ignored]

        parameters = sorted([*p1, *[p.replace(kind=Parameter.KEYWORD_ONLY) for p in p2]], key=lambda p: p.kind)
        wrapper.__signature__ = signature(func).replace(parameters=parameters)
        return wrapper
    
    return decorator

def entry_point(*components: Union[Callable, Iterable[Callable]], ignored: Optional[Iterable[str]] = None):
    ignored = set(ignored) if ignored is not None else set()

    def decorator(func):
        for sub_func in components:
            c_ignores = getattr(sub_func, IGNORED_KEYWORDS_ON_ENTRY_POINT_ATTR_NAME, set()) | ignored
            func = use(sub_func, c_ignores)(func)
        return func
    
    return decorator
