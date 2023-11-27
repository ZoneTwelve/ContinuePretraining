import os
from typing import Any, Optional

import torch
from lightning.pytorch.utilities.deepspeed import \
    convert_zero_checkpoint_to_fp32_state_dict


def get_dtype(client_state: dict[str, Any]):
    if client_state['ds_config'].get('fp16', {}).get('enabled', False):
        return torch.half
    
    if client_state['ds_config'].get('bf16', {}).get('enabled', False):
        return torch.bfloat16
    
    return torch.float


def get_lightning_checkpoint_from_zero_checkpoint(checkpoint_dir: str, tag: Optional[str] = None):
    client_state = convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, os.devnull, tag)
    dtype = get_dtype(client_state)
    for v in client_state['state_dict'].values():
        v.data = v.data.to(dtype)
    return client_state
