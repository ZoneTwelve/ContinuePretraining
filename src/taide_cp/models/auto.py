from os import PathLike

from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

__all__ = ['AutoConfig', 'AutoModelForCausalLM', 'AutoTokenizer']

class AutoConfig(AutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike, **kwargs) -> PretrainedConfig:
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.pop('model_type', None) == 'mpt':
            from ..models.mpt import MPTConfig

            kwargs.pop('trust_remote_code', None)
            return MPTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_config(cls, config: PretrainedConfig, **kwargs):
        if config.model_type == 'mpt':
            from ..models.mpt import MPTForCausalLM
            kwargs.pop('trust_remote_code', None)
            return MPTForCausalLM._from_config(config, **kwargs)

        return super().from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike, *model_args, **kwargs) -> PreTrainedModel:
        config = kwargs.pop('config', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)

        org_config = config
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        if config.model_type == 'mpt':
            from ..models.mpt import MPTForCausalLM

            kwargs.pop('trust_remote_code', None)
            return MPTForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

        return super().from_pretrained(pretrained_model_name_or_path, config=org_config, *model_args, **kwargs)
    

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
