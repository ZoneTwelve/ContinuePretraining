import shutil
from contextlib import contextmanager
from typing import Any, Dict

import lightning as L
import torch
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies.deepspeed import (DeepSpeedStrategy,
                                                    warning_cache)
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers.deepspeed import HfDeepSpeedConfig

from ..utils.deepspeed import get_lightning_checkpoint_from_zero_checkpoint


class DeepSpeedSkippedStepsCallback(Callback):
    def __init__(self, raise_error_at_min_scale: bool | None = None) -> None:
        super().__init__()

        self.raise_error_at_min_scale = raise_error_at_min_scale

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        assert isinstance(trainer.strategy, DeepSpeedStrategy)
        loss_scaler = trainer.strategy.deepspeed_engine.optimizer.loss_scaler
        if self.raise_error_at_min_scale is not None:
            loss_scaler.raise_error_at_min_scale = self.raise_error_at_min_scale

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        assert isinstance(trainer.strategy, DeepSpeedStrategy)
        trainer.progress_bar_metrics['skipped_steps'] = trainer.strategy.deepspeed_engine.skipped_steps


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
    def offload_optimizer(self) -> bool:
        assert isinstance(self.config, dict)
        zero_optimization = self.config.get('zero_optimization', {})
        return 'offload_optimizer' in zero_optimization

    @property
    def ds_init_context(self):
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

    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH, storage_options: Any | None = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        weights_only = 'optimizer_states' not in checkpoint

        # broadcast the filepath from rank 0 to ensure all the states are saved in a common filepath
        filepath = self.broadcast(filepath)

        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used."
            )

        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            warning_cache.warn(
                "When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint", exclude_frozen_parameters=True)

        if self.is_global_zero and weights_only:
            checkpoint = get_lightning_checkpoint_from_zero_checkpoint(filepath, dtype=self.ds_dtype)
            shutil.rmtree(filepath)
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options)
