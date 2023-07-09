from transformers import GPT2TokenizerFast, OPTConfig, OPTForCausalLM

from ...models.opt import DeepSpeedOPTForCausalLM
from .lightning_module_for_pre_training import LightningModuleForPreTraining


class OPTLightningModuleForPreTraining(LightningModuleForPreTraining):
    name = 'opt'
    model_class = OPTForCausalLM
    ds_model_class = DeepSpeedOPTForCausalLM
    config_class = OPTConfig
    tokenizer_class = GPT2TokenizerFast
