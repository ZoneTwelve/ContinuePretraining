import dataclasses
from types import UnionType
from typing import ClassVar

from typing_extensions import Self, dataclass_transform


def _from_dict(cls, x):
    if not isinstance(x, dict):
        return x

    if isinstance(cls, UnionType):
        for c in cls.__args__:
            if dataclasses.is_dataclass(c):
                cls = c
                break
        else:
            return x
    
    if not dataclasses.is_dataclass(cls):
        return x
    
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    return cls(**{f: _from_dict(types[f], x[f]) for f in x})


class IgnoredValue:
    def __new__(cls) -> None:
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

@dataclass_transform()
class ConfigMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        new_cls = super().__new__(cls, name, bases, attrs)
        return dataclasses.dataclass(new_cls)


class ConfigBase(metaclass=ConfigMeta):
    _keys_to_ignore_on_log: ClassVar[list[str]] = []

    def __post_init__(self): ...

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, name):
        return getattr(self, name)
    
    def __setitem__(self, name, value):
        setattr(self, name, value)

    def to_dict(self):
        return dataclasses.asdict(self)

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
    
    def get_config_to_log(self):
        return self.replace(**{k: IgnoredValue() for k in self._keys_to_ignore_on_log})
    
    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return _from_dict(cls, d)
