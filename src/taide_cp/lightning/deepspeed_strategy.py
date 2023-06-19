import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict

import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers.deepspeed import HfDeepSpeedConfig

if TYPE_CHECKING:
    from lightning.fabric.utilities.types import _PATH

from ..utils.deepspeed import get_lightning_checkpoint_from_zero_checkpoint


class DeepSpeedStrategy(DeepSpeedStrategy):
    @property
    def ds_dtype(self):
        dtype_mapping = {
            '16-mixed': torch.float16,
            'bf16-mixed': torch.bfloat16,
        }
        dtype = dtype_mapping.setdefault(self.precision_plugin.precision, torch.float32)
        return dtype

    @property
    def is_using_offload(self) -> bool:
        assert isinstance(self.config, dict)
        zero_optimization = self.config.get('zero_optimization')
        return (
            zero_optimization is not None
            and (
                'offload_optimizer' in zero_optimization
                or 'offload_param' in zero_optimization
            )
        )

    @property
    def deepspeed_init_context(self):
        if self.zero_stage_3:
            import deepspeed

            return deepspeed.zero.Init(
                remote_device=self.remote_device,
                pin_memory=True,
                config_dict_or_path=self.config,
                dtype=self.ds_dtype
            )
        return self.model_sharded_context()
    
    def setup_environment(self) -> None:
        super().setup_environment()
        self._hf_ds_config = HfDeepSpeedConfig(self.config)
    
    @contextmanager
    def model_sharded_context(self):
        yield

    def save_checkpoint(self, checkpoint: Dict, filepath: "_PATH", storage_options: Any | None = None) -> None:
        weights_only = 'optimizer_states' not in checkpoint        
        super().save_checkpoint(checkpoint, filepath, storage_options)

        if self.is_global_zero and weights_only:
            checkpoint = get_lightning_checkpoint_from_zero_checkpoint(filepath, dtype=self.ds_dtype)
            shutil.rmtree(filepath)
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options)
