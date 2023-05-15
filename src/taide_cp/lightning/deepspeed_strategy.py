from contextlib import contextmanager

import deepspeed
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers.deepspeed import HfDeepSpeedConfig


class DeepSpeedStrategy(DeepSpeedStrategy):
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
        dtype_mapping = {
            '16-mixed': torch.float16,
            'bf16-mixed': torch.bfloat16,
        }
        dtype = dtype_mapping.setdefault(self.precision_plugin.precision, torch.float32)

        if self.zero_stage_3:
            return deepspeed.zero.Init(
                remote_device=self.remote_device,
                pin_memory=True,
                config_dict_or_path=self.config,
                dtype=dtype
            )
        return self.model_sharded_context()
    
    def setup_environment(self) -> None:
        super().setup_environment()
        self._hf_ds_config = HfDeepSpeedConfig(self.config)
    
    @contextmanager
    def model_sharded_context(self):
        yield
