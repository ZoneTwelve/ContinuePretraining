import os
from typing import Any, Dict, Mapping, Optional

import torch
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.checkpoint import (FileSystemReader, FileSystemWriter,
                                          load_state_dict, save_state_dict)
from torch.distributed.checkpoint.optimizer import \
    load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from torch.optim import Optimizer

CHECKPOINT_FILE = 'checkpoint.pt'
STATE_DICT_SUBDIR = 'state_dict'


class FSDPStrategy(FSDPStrategy):
    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return True
    
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
        state_dict = {
            'state_dict': checkpoint.pop('state_dict'),
            'optimizer_states': checkpoint.pop('optimizer_states')
        }
        
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(
                checkpoint,
                os.path.join(filepath, CHECKPOINT_FILE),
                storage_options=storage_options
            )
        
        save_state_dict(state_dict, FileSystemWriter(os.path.join(filepath, STATE_DICT_SUBDIR)))

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        checkpoint = self.checkpoint_io.load_checkpoint(os.path.join(checkpoint_path, CHECKPOINT_FILE))
        storage_reader = FileSystemReader(os.path.join(checkpoint_path, STATE_DICT_SUBDIR))

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