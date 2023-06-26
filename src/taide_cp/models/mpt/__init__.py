from transformers import AutoModelForCausalLM, AutoConfig

from .configuration_mpt import MPTConfig
from .deepspeed_mpt import DeepSpeedMPTForCausalLM, DeepSpeedMPTModel
from .modeling_mpt import MPTForCausalLM, MPTModel

AutoConfig.register('mpt', MPTConfig)
AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)
