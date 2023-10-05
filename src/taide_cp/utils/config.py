import dataclasses

from typing_extensions import dataclass_transform


@dataclass_transform()
class ConfigMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        new_cls = super().__new__(cls, name, bases, attrs)
        return dataclasses.dataclass(new_cls)


class ConfigBase(metaclass=ConfigMeta):
    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, name):
        return getattr(self, name)
    
    def __setitem__(self, name, value):
        setattr(self, name, value)

    def asdict(self):
        return dataclasses.asdict(self)
