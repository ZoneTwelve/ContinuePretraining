from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from .causal_lm import LitCausalLM


class LitLlamaForCausalLM(LitCausalLM):
    model_class = LlamaForCausalLM
    config_class = LlamaConfig
    tokenizer_class = LlamaTokenizer

    model: LlamaForCausalLM
    model_config: LlamaConfig
