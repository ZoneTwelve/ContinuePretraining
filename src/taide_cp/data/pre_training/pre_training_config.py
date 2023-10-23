from enum import auto

from transformers import PreTrainedTokenizerBase

from ...utils.str_enum import StrEnum
from ..datamodule import DataModuleConfig


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    CONCAT_AND_TRUNCATE = auto()
    GROUP_BY_LENGTH = auto()


class PreTrainingConfig(DataModuleConfig):
    _keys_to_ignore_on_log = ['tokenizer']

    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None
    stride: int | None = None
    concat_method: ConcatMethod | str = ConcatMethod.CONCAT_AND_TRUNCATE
    pad_to_multiple_of: int | None = None

    def __post_init__(self):
        super().__post_init__()

        self.stride = self.stride or self.max_length
        self.concat_method = ConcatMethod(self.concat_method)
