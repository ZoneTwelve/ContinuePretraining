from os import PathLike

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..models.mpt import MPTConfig, MPTForCausalLM

__all__ = ['AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer']


AutoConfig.register('mpt', MPTConfig)
AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)


class AutoTokenizer(AutoTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike, *inputs, **kwargs) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        from transformers.models.auto.tokenization_auto import \
            get_tokenizer_config
        
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        tokenizer_class = tokenizer_config['tokenizer_class']

        if tokenizer_class == 'LlamaTokenizer':
            kwargs['use_fast'] = kwargs.pop('use_fast', False)
        
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
