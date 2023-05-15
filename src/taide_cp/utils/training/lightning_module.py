
import os
from inspect import signature

import lightning as L
from lightning.pytorch.loggers import WandbLogger


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
