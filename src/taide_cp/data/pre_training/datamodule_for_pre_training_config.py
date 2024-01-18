from dataclasses import field
from enum import auto

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ...utils.str_enum import StrEnum
from ..datamodule import DataModuleConfig


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    CONCAT_AND_TRUNCATE = auto()


class DataModuleForPreTrainingConfig(DataModuleConfig):
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    max_length: int | None = None
    stride: int | None = None
    concat_method: ConcatMethod | str = ConcatMethod.CONCAT_AND_TRUNCATE
    pad_to_multiple_of: int | None = None
    sample_rate: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.concat_method = ConcatMethod(self.concat_method.lower())

        if self.concat_method == ConcatMethod.CONCAT_AND_TRUNCATE:
            assert self.max_length is not None, f"You must set `max_length` to use `CONCAT_AND_TRUNCATE`"

        assert self.stride is None or self.max_length is not None, "You must also set `max_length` to use `stride`"
