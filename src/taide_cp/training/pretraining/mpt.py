import torch
from transformers import GPTNeoXTokenizerFast

from ...models.mpt import DeepSpeedMPTForCausalLM, MPTConfig, MPTForCausalLM
from ...models.mpt.blocks import MPTBlock
from .lightning_module_for_pre_training import LightningModuleForPreTraining


class MPTLightningModuleForPreTraining(LightningModuleForPreTraining):
    model_class = MPTForCausalLM
    ds_model_class = DeepSpeedMPTForCausalLM
    config_class = MPTConfig
    tokenizer_class = GPTNeoXTokenizerFast
    
    config: MPTConfig

    fsdp_transformer_layer_cls = {MPTBlock}

    def on_before_configure_sharded_model(self):
        self.config.init_device = torch.device('cuda', self.local_rank)
        self.config.max_seq_len = self.max_length
