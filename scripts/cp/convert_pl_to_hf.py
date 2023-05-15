import functools
from typing import Any, Dict, Optional, cast

import fire
import torch
from accelerate import init_empty_weights
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from taide_cp.utils.deepspeed import get_state_dict_from_zero_checkpoint


def rgetattr(obj: Any, attr: str):
    return functools.reduce(getattr, [obj] + attr.split('.'))

def rsetattr(obj: Any, attr: str, val: Any):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def patch_state_dict(state_dict: Dict[str, torch.Tensor]):
    return {k.removeprefix('_forward_module.model.'): v for k, v in state_dict.items()}

def load_state_dict(model: PreTrainedModel, state_dict: Dict[str, torch.Tensor]):
    for k, v in state_dict.items():
        rsetattr(model, k, torch.nn.Parameter(v))
    model.tie_weights()
    return model

def main(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or config_path, use_fast=True)
    config = AutoConfig.from_pretrained(config_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        model = cast(PreTrainedModel, model)

    if tokenizer_path is not None and model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    state_dict = get_state_dict_from_zero_checkpoint(checkpoint_path, dtype=torch.half)
    state_dict = patch_state_dict(state_dict)
    model = load_state_dict(model, state_dict)
    
    model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
