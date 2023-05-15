from abc import ABC, abstractmethod
from typing import Any, Dict, List

from torch import Tensor
from transformers import PreTrainedTokenizerBase


class DataCollator(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, batch: List[Any]) -> Dict[str, Tensor]:
        pass
    
    @staticmethod
    def convert_list_of_dict_to_dict_of_list(list_of_dict: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        return {k: [dic[k] for dic in list_of_dict] for k in list_of_dict[0]}
