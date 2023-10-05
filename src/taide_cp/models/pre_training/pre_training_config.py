from dataclasses import MISSING, field
from enum import auto
from typing import Literal

from peft import LoraConfig, TaskType

from ...utils import StrEnum
from ...utils.config import ConfigBase
from ..lr_schedulers import LearningRateSchedulerType


__all__ = [
    'InitializingStrategy',
    'TrainingStrategy',
    'ExtendVocabConfig',
    'OptimizerConfig',
    'PreTrainingConfig',
    'PreTrainingWithLoRAConfig'
]


class InitializingStrategy(StrEnum):
    DEFAULT = auto()
    MEAN = auto()
    SAMPLE = auto()


class TrainingStrategy(StrEnum):
    FULL = auto()
    NEW_TOKENS_ONLY = auto()
    ALL_TOKENS = auto()


class ExtendVocabConfig(ConfigBase):
    initializing_strategy: InitializingStrategy = InitializingStrategy.SAMPLE
    training_strategy: TrainingStrategy = TrainingStrategy.FULL


class OptimizerConfig(ConfigBase):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5
    weight_decay: float = 1e-1
    lr_scheduler_type: LearningRateSchedulerType = LearningRateSchedulerType.NONE
    num_warmup_steps: int = 0
    min_lr_factor: float = 0.1


class PreTrainingConfig(ConfigBase):
    model_path: str
    revision: str = 'main'
    tokenizer_path: str | None = None
    extend_vocab: Literal[False] | ExtendVocabConfig = False
    max_position_embeddings: int | None = None
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        self.tokenizer_path = self.tokenizer_path or self.model_path


class PreTrainingWithLoRAConfig(PreTrainingConfig):
    lora_config: LoraConfig | None = None
    lora_path: str | None = None
    quantization_config: dict | None = None

    def __post_init__(self):
        super().__post_init__()

        assert self.lora_config or self.lora_path
        
        if self.lora_config:
            self.lora_config.task_type = TaskType.CAUSAL_LM
