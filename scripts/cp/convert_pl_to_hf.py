import os
from typing import Dict, Optional, cast

import fire
import torch
from accelerate import init_empty_weights
from transformers import PreTrainedModel

from taide_cp.models import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from taide_cp.utils import rsetattr
from taide_cp.utils.deepspeed import \
    get_lightning_checkpoint_from_zero_checkpoint


def patch_state_dict(state_dict: Dict[str, torch.Tensor]):
    return {k.removeprefix('model.'): v for k, v in state_dict.items()}

def patch_partial_embeddings(model: PreTrainedModel, state_dict: Dict[str, torch.Tensor]):
    new_state_dict: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.endswith('.w1'):
            continue

        if k.endswith('.w2'):
            new_key = k.replace('.w2', '.weight')
            old_embeddings = model.get_parameter(new_key).data
            num_old_embeddings = old_embeddings.size(0)
            num_embeddings = num_old_embeddings + v.size(0)
            embedding_size = v.size(1)
            x = torch.empty(num_embeddings, embedding_size)
            x[:num_old_embeddings].copy_(old_embeddings)
            x[num_old_embeddings:].copy_(v)
            k, v = new_key, x
        new_state_dict[k] = v
    return new_state_dict

def load_state_dict(model: PreTrainedModel, state_dict: Dict[str, torch.Tensor]):
    for k, v in state_dict.items():
        rsetattr(model, k, torch.nn.Parameter(v))
    model.tie_weights()
    return model

def main(
    checkpoint_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
):
    if os.path.isdir(checkpoint_path):
        checkpoint = get_lightning_checkpoint_from_zero_checkpoint(checkpoint_path, dtype=torch.half)
    else:
        checkpoint = torch.load(checkpoint_path, 'cpu')

    state_dict = patch_state_dict(checkpoint['state_dict'])
    
    hyper_parameters = checkpoint['hyper_parameters']

    model_path = model_path or hyper_parameters['model_path']
    tokenizer_path = tokenizer_path or hyper_parameters['tokenizer_path'] or model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model = cast(PreTrainedModel, model)

    if hyper_parameters['extend_tokens']:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True, config=config)
        model = cast(PreTrainedModel, model)
        state_dict = patch_partial_embeddings(model, state_dict)
        model.resize_token_embeddings(len(tokenizer))

    model = load_state_dict(model, state_dict)
    model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
