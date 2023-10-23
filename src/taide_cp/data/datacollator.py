from abc import ABC, abstractmethod
from typing import Any


class DataCollator(ABC):
    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]: ...
