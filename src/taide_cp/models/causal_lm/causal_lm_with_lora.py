from .causal_lm_config import LitCausalLMWithLoRAConfig
from .causal_lm import LitCausalLM
from peft import PeftModelForCausalLM


class LitCausalLMWithLoRA(LitCausalLM):
    config: LitCausalLMWithLoRAConfig
    model: PeftModelForCausalLM

    def __init__(self, config: LitCausalLMWithLoRAConfig) -> None:
        super().__init__(config)

        if config.quantization_config is not None:
            if not hasattr(self.model_config, 'quantization_config'):
                self.model_config.quantization_config = {}
            self.model_config.quantization_config |= config.quantization_config

    def configure_model(self) -> None:
        super().configure_model()

        if self.config.lora_config:
            self.model = PeftModelForCausalLM(self.model, self.config.lora_config)
        else:
            self.model = PeftModelForCausalLM.from_pretrained(self.model, self.config.lora_path, is_trainable=True)

        self.model.base_model.enable_input_require_grads()
