from typing import Tuple

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

from ...models.llama import DeepSpeedLlamaForCausalLM
from .lightning_module_for_pre_training import LightningModuleForPreTraining


class LLaMALightningModuleForPreTraining(LightningModuleForPreTraining):
    model_class = LlamaForCausalLM
    ds_model_class = DeepSpeedLlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizerFast

    def __postinit__(self) -> None:
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.bos_token})
