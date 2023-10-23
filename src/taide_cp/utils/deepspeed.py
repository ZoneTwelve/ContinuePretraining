from typing import Optional

import torch
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint, get_model_state_file,
    get_optim_files)
from lightning.pytorch.utilities.deepspeed import CPU_DEVICE, ds_checkpoint_dir, convert_zero_checkpoint_to_fp32_state_dict

__all__ = ['get_lightning_checkpoint_from_zero_checkpoint']

def get_lightning_checkpoint_from_zero_checkpoint(checkpoint_dir: str, tag: Optional[str] = None, dtype: torch.dtype = torch.float32):
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

    # additional logic to ensure we keep the lightning state dict as well from rank 0.
    deepspeed_states = [
        "module",
        "optimizer",
        "lr_scheduler",
        "csr_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "dp_world_size",
        "mp_world_size",
    ]
    checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    model_file = get_model_state_file(checkpoint_dir, zero_stage)
    client_state = torch.load(model_file, map_location=CPU_DEVICE)
    client_state = {key: value for key, value in client_state.items() if key not in deepspeed_states}
    # State dict keys will include reference to wrapper _LightningModuleWrapperBase
    # Delete `module` prefix before saving.
    state_dict = {k.partition("module.")[2]: v.to(dtype) for k, v in state_dict.items()}
    client_state["state_dict"] = state_dict
    return client_state
