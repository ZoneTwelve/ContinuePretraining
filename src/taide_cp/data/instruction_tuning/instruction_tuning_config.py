from enum import auto
from string import Template

from transformers import PreTrainedTokenizerBase

from ...utils.str_enum import StrEnum
from ..datamodule import DataModuleConfig


class OverlongHandlingMethod(StrEnum):
    DROP = auto()
    TRUNCATE = auto()


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    GROUP_BY_LENGTH = auto()


class InstructionTuningConfig(DataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    prompt_template: Template | str
    response_template: Template | str
    max_length: int | None = None
    overlong_handling_method: OverlongHandlingMethod | str = OverlongHandlingMethod.DROP
    concat_method: ConcatMethod | str = ConcatMethod.NO_CONCAT
    pad_to_multiple_of: int | None = None

    def __post_init__(self):
        super().__post_init__()

        self.prompt_template = Template(self.prompt_template) if isinstance(self.prompt_template, str) else self.prompt_template
        self.response_template = Template(self.response_template) if isinstance(self.response_template, str) else self.response_template
        self.overlong_handling_method = OverlongHandlingMethod(self.overlong_handling_method)
        self.concat_method = ConcatMethod(self.concat_method)
