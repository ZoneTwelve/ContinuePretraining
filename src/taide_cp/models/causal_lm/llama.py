from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from .causal_lm import LitCausalLM, LitCausalLMConfig


class LitLlamaForCausalLM(LitCausalLM):
    model_class = LlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizer

    model: LlamaForCausalLM
    model_config: LlamaConfig

    def __init__(self, config: LitCausalLMConfig) -> None:
        super().__init__(config)

        if self.config.max_position_embeddings is not None:
            self.model_config.max_position_embeddings = self.config.max_position_embeddings
