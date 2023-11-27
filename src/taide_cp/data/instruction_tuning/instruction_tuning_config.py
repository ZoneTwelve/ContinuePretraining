from enum import auto

from transformers import PreTrainedTokenizerBase

from ...utils.str_enum import StrEnum
from ..datamodule import DataModuleConfig
from ..template import TemplateBase


class OverlongHandlingMethod(StrEnum):
    DROP = auto()
    TRUNCATE = auto()


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    GROUP_BY_LENGTH = auto()


class InstructionTuningConfig(DataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    prompt_template: TemplateBase | str
    response_template: TemplateBase | str
    max_length: int | None = None
    overlong_handling_method: OverlongHandlingMethod | str = OverlongHandlingMethod.DROP
    concat_method: ConcatMethod | str = ConcatMethod.NO_CONCAT
    pad_to_multiple_of: int | None = None

    def __post_init__(self):
        super().__post_init__()

        self.prompt_template = TemplateBase(self.prompt_template) if isinstance(self.prompt_template, str) else self.prompt_template
        self.response_template = TemplateBase(self.response_template) if isinstance(self.response_template, str) else self.response_template
        self.overlong_handling_method = OverlongHandlingMethod(self.overlong_handling_method)
        self.concat_method = ConcatMethod(self.concat_method)
