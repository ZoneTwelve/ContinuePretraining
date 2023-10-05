import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.debug_utils import DebugUnderflowOverflow, get_abs_min_max
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from ...patchers import Patcher
from .lightning_module_for_pre_training import *
from .pre_training_config import *


def detect_nan(var, ctx):
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    # if torch.isinf(var).any().item():
    #     detected = True
    #     print(f"{ctx} has infs")
    return detected


class DebugNaN(DebugUnderflowOverflow):
    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):
            self.expand_frame(get_abs_min_max(var, ctx))
            if detect_nan(var, ctx):
                self.detected_overflow = True
        elif var is None:
            self.expand_frame(f"{'None':>17} {ctx}")
        else:
            self.expand_frame(f"{'not a tensor':>17} {ctx}")

class LLaMAForPreTraining(LightningModuleForPreTraining):
    model_class = LlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizer

    model: LlamaForCausalLM
    model_config: LlamaConfig

    fsdp_transformer_layer_cls = LlamaDecoderLayer

    def __init__(self, config: PreTrainingConfig, patchers: Patcher | list[Patcher] | None = None) -> None:
        super().__init__(config, patchers)

        if self.config.max_position_embeddings is not None:
            self.model_config.max_position_embeddings = self.config.max_position_embeddings


class LLaMAForPreTrainingWithLoRA(LightningModuleForPreTrainingWithLoRA):
    model_class = LlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizer

    model: LlamaForCausalLM
    model_config: LlamaConfig

    def __init__(self, config: PreTrainingWithLoRAConfig, patchers: Patcher | list[Patcher] | None = None) -> None:
        super().__init__(config, patchers)

        if self.config.max_position_embeddings is not None:
            self.model_config.max_position_embeddings = self.config.max_position_embeddings
