from abc import ABC, abstractmethod
from typing import Any

from .datamodule_config import DataModuleConfig


class DataCollator(ABC):
    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()

        self.config = config

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]: ...
