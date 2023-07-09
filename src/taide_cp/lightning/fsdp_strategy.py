from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type

import torch
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.checkpoint import (FileSystemReader, FileSystemWriter,
                                          load_state_dict, save_state_dict)
from torch.distributed.checkpoint.optimizer import \
    load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.nn import Module
from torch.optim import Optimizer


class FSDPStrategy(FSDPStrategy):
    def __init__(self, accelerator: Any | None = None, parallel_devices: List[torch.device] | None = None, cluster_environment: ClusterEnvironment | None = None, checkpoint_io: CheckpointIO | None = None, precision_plugin: PrecisionPlugin | None = None, process_group_backend: str | None = None, cpu_offload: bool | Any | None = None, mixed_precision: Any | None = None, activation_checkpointing: Type[Module] | List[Type[Module]] | None = None, **kwargs: Any) -> None:
        super().__init__(accelerator, parallel_devices, cluster_environment, checkpoint_io, precision_plugin, process_group_backend, cpu_offload, mixed_precision, activation_checkpointing, **kwargs)


    def lightning_module_state_dict(self) -> Dict[str, Any]:
        with FSDP.state_dict_type(
            self.lightning_module,
            StateDictType.SHARDED_STATE_DICT,
        ):
            return self.lightning_module.state_dict()
        
    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        with FSDP.state_dict_type(self.lightning_module, StateDictType.SHARDED_STATE_DICT):
            self.lightning_module.load_state_dict(checkpoint['state_dict'])

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, torch.Tensor]:
        with FSDP.state_dict_type(
            self.lightning_module,
            StateDictType.SHARDED_STATE_DICT,
        ):
            return FSDP.optim_state_dict(self.lightning_module, optimizer)
    
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        with FSDP.state_dict_type(
            self.lightning_module,
            StateDictType.SHARDED_STATE_DICT,
        ):
            for optimizer, opt_state in zip(self.optimizers, checkpoint['optimizer_states']):
                opt_state = FSDP.optim_state_dict_to_load(opt_state, self.lightning_module, optimizer)
                optimizer.load_state_dict(opt_state)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        dirpath = Path(filepath).with_suffix('')

        state_dict = {
            'state_dict': checkpoint.pop('state_dict'),
            'optimizer_states': checkpoint.pop('optimizer_states')
        }
        
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(
                checkpoint,
                dirpath.joinpath(f'checkpoint.ckpt'),
                storage_options=storage_options
            )
        
        save_state_dict(state_dict, FileSystemWriter(dirpath.joinpath('state_dict')))

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        dirpath = Path(checkpoint_path).with_suffix('')

        checkpoint = self.checkpoint_io.load_checkpoint(dirpath.joinpath(f'checkpoint.ckpt'))
        storage_reader = FileSystemReader(dirpath.joinpath('state_dict'))
        
        with FSDP.state_dict_type(
            self.lightning_module,
            StateDictType.SHARDED_STATE_DICT
        ):
            state_dict = {
                'state_dict': self.lightning_module.state_dict(),
            }

            load_state_dict(state_dict, storage_reader)
            
            state_dict |= load_sharded_optimizer_state_dict(
                model_state_dict=state_dict['state_dict'],
                optimizer_key='optimizer_states',
                storage_reader=storage_reader,
            )
        
        checkpoint |= state_dict
        return checkpoint