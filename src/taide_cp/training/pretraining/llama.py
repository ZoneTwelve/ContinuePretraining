from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from ...models.llama import DeepSpeedLlamaForCausalLM
from .lightning_module_for_pre_training import LightningModuleForPreTraining


class LLaMALightningModuleForPreTraining(LightningModuleForPreTraining):
    name = 'llama'
    model_class = LlamaForCausalLM
    ds_model_class = DeepSpeedLlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizer

    config: LlamaConfig

    def on_after_init(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.bos_token})

        self.config.max_position_embeddings = self.max_length
