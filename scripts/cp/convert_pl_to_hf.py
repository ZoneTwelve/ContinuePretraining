from typing import Dict, Optional, cast

import fire
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)
from transformers.modeling_utils import no_init_weights

from taide_cp.utils.deepspeed import get_state_dict_from_zero_checkpoint


def patch_state_dict(state_dict: Dict[str, torch.Tensor]):
    for k in list(state_dict.keys()):
        state_dict[k.removeprefix('_forward_module.model.')] = state_dict.pop(k)

def main(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or config_path, use_fast=True)
    config = AutoConfig.from_pretrained(config_path)
    with no_init_weights():
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_config(config)
        model = cast(PreTrainedModel, model)
        torch.set_default_dtype(torch.float)

    if tokenizer_path is not None and model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(checkpoint_path, 'cpu')
    patch_state_dict(state_dict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(incompatible_keys.missing_keys) - set(model._keys_to_ignore_on_load_missing or [])
    assert not missing_keys
    model.tie_weights()

    model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
