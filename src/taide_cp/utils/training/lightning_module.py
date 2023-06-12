
import os
from inspect import signature
from typing import Dict

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys


def _state_dict_hook(module: nn.Module, state_dict: Dict[str, Tensor], prefix: str, local_metadata):
    trainables = set(n for n, p in module.named_parameters() if p.requires_grad)
    for k in list(state_dict.keys()):
        if k not in trainables:
            state_dict.pop(k)

def _load_state_dict_post_hook(module: nn.Module, incompatible_keys: _IncompatibleKeys):
    missing_keys = set(incompatible_keys.missing_keys) & set(n for n, p in module.named_parameters() if p.requires_grad)
    incompatible_keys.missing_keys[:] = list(missing_keys)


class LightningModuleX(L.LightningModule):
    logger: WandbLogger

    @property
    def checkpoint_dir(self):
        return os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, 'checkpoints')
    
    @property
    def is_global_zero(self):
        return self.trainer.is_global_zero
    
    @property
    def strategy(self):
        return self.trainer.strategy
    
    def __init__(self) -> None:
        super().__init__()

        self._register_state_dict_hook(_state_dict_hook)
        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

    def __init_subclass__(cls) -> None:
        __init__ = cls.__init__
        
        def __new_init__(self, *args, **kwargs):
            r = __init__(self, *args, **kwargs)
            cls.__postinit__(self)
            return r
        
        __new_init__.__signature__ = signature(__init__)
        
        cls.__init__ = __new_init__

    def __postinit__(self) -> None:
        pass

    def print_rank0(self, *args, **kwargs) -> None:
        if self.is_global_zero:
            print(*args, **kwargs)
