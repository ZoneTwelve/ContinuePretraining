import inspect
import os
from functools import wraps
from typing import Dict, Literal

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

def hook(target: str, type: Literal['pre', 'post']):
    def decorator(hook):
        setattr(hook, '__hook__', dict(target=target, type=type))
        return hook
    return decorator

def enable_hooks(cls):
    def hook_function(hooked, pre, post):
        @wraps(hooked)
        def wrapper(self, *args, **kwargs):
            if pre:
                pre(self)
            r = hooked(self, *args, **kwargs)
            if post:
                post(self)
            return r
        return wrapper
    
    @classmethod
    def __init_subclass__(subclass):
        mapping = {}
        for name, hook in inspect.getmembers(cls, lambda x: hasattr(x, '__hook__')):
            meta = getattr(hook, '__hook__')
            x = mapping.setdefault(meta['target'], {})
            x[meta['type']] = name

        for target, hooks in mapping.items():
            hooked = getattr(subclass, target)
            pre = getattr(subclass, hooks['pre'], None) if 'pre' in hooks else None
            post = getattr(subclass, hooks['post'], None) if 'post' in hooks else None
            setattr(subclass, target, hook_function(hooked, pre, post))

    cls.__init_subclass__ = __init_subclass__
    return cls

@enable_hooks
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
  
    def print_rank0(self, *args, **kwargs) -> None:
        if self.is_global_zero:
            print(*args, **kwargs)

    @hook('__init__', 'post')
    def __postinit__(self): ...

    @hook('configure_sharded_model', 'pre')
    def on_before_configure_sharded_model(self): ...
