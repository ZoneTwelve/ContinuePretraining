from contextlib import AbstractContextManager
from types import TracebackType

import torch
from safetensors import safe_open


class _TensorSlice:
    @property
    def shape(self):
        return self.tensor_slice.get_shape()
    
    def __init__(self, tensor_slice) -> None:
        self.tensor_slice = tensor_slice

    def __getitem__(self, index: slice | tuple[slice]):
        return self.tensor_slice[index]

    def get_shape(self):
        return self.tensor_slice.get_shape()


class SafeTensors(AbstractContextManager):
    def __init__(self, filename: str, framework: str = 'pt', device: str = 'cpu') -> None:
        self.f = safe_open(filename, framework, device)

    def __enter__(self):
        return self
    
    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        return self.f.__exit__(__exc_type, __exc_value, __traceback)

    def keys(self) -> list[str]:
        return self.f.keys()
    
    def get_tensor(self, key: str) -> torch.Tensor:
        return self.f.get_tensor(key)
    
    def get_slice(self, key: str) -> _TensorSlice:
        return _TensorSlice(self.f.get_slice(key))
