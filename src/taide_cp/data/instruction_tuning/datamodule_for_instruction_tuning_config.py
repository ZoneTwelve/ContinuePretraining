import logging
from enum import auto
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from taide_cp.data import DataModuleConfig
from taide_cp.utils.str_enum import StrEnum

from .template import PREDEFINED_TEMPLATES

logger = logging.getLogger(__name__)


class OverlongHandlingMethod(StrEnum):
    DROP = auto()
    TRUNCATE = auto()


class ConcatMethod(StrEnum):
    NO_CONCAT = auto()
    GROUP_BY_LENGTH = auto()


class DataModuleForInstructionTuningConfig(DataModuleConfig):
    tokenizer: PreTrainedTokenizerBase
    chat_template: str | None = None
    max_length: int | None = None
    overlong_handling_method: OverlongHandlingMethod | str = OverlongHandlingMethod.DROP
    concat_method: ConcatMethod | str = ConcatMethod.NO_CONCAT
    pad_to_multiple_of: int | None = None
    empty_system_prompt_rate: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.chat_template is None:
            logger.info(f'`chat_template` is not set, default template of the tokenizer will be used.')
        elif Path(self.chat_template).exists():
            logger.info(f'Found template file at `{self.chat_template}`.')
            with open(self.chat_template) as f:
                self.chat_template = f.read()
        elif self.chat_template in PREDEFINED_TEMPLATES:
            logger.info(f'Using predefined template `{self.chat_template}`.')
            self.chat_template = PREDEFINED_TEMPLATES[self.chat_template]
        else:
            logger.info('Treat `chat_template` as a template directly.')

        self.overlong_handling_method = OverlongHandlingMethod(self.overlong_handling_method)
        self.concat_method = ConcatMethod(self.concat_method)
