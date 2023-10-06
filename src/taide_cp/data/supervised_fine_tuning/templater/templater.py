from abc import ABC, abstractmethod


class Templater(ABC):
    @abstractmethod
    def apply(self, **kwargs) -> tuple[str, str]: ...

    def match(self, **kwargs) -> bool:
        return True
