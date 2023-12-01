import logging
import shutil
from contextlib import contextmanager
from typing import Any, Dict, List

import lightning as L
import torch
from lightning.fabric.plugins import ClusterEnvironment
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.plugins import AsyncCheckpointIO, Precision
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ..utils.deepspeed import get_lightning_checkpoint_from_zero_checkpoint


class EnhancedDeepSpeedStrategy(DeepSpeedStrategy):

    def __init__(
        self, 
        accelerator: Accelerator | None = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str | None = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = 'cpu',
        nvme_path: str = '/local_nvme',
        params_buffer_count: int = 5,
        params_buffer_size: int = 100000000,
        max_in_cpu: int = 1000000000,
        offload_optimizer_device: str = 'cpu',
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1000000000000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200000000,
        reduce_bucket_size: int = 200000000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: str | int = 'auto',
        config: _PATH | Dict[str, Any] | None = None,
        logging_level: int | str = logging.WARN,
        parallel_devices: List[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        exclude_frozen_parameters: bool = False,
        raise_error_at_min_scale: bool | None = None,
    ):
        if isinstance(logging_level, str):
            logging_level = getattr(logging, logging_level.upper())

        super().__init__(accelerator, zero_optimization, stage, remote_device, offload_optimizer, offload_parameters, offload_params_device, nvme_path, params_buffer_count, params_buffer_size, max_in_cpu, offload_optimizer_device, optimizer_buffer_count, block_size, queue_depth, single_submit, overlap_events, thread_count, pin_memory, sub_group_size, contiguous_gradients, overlap_comm, allgather_partitions, reduce_scatter, allgather_bucket_size, reduce_bucket_size, zero_allow_untested_optimizer, logging_batch_size_per_gpu, config, logging_level, parallel_devices, cluster_environment, loss_scale, initial_scale_power, loss_scale_window, hysteresis, min_loss_scale, partition_activations, cpu_checkpointing, contiguous_memory_optimization, synchronize_checkpoint_boundary, load_full_weights, precision_plugin, process_group_backend)
        
        self.exclude_frozen_parameters = exclude_frozen_parameters
        self.raise_error_at_min_scale = raise_error_at_min_scale

    @property
    def offload_optimizer(self) -> bool:
        assert isinstance(self.config, dict)
        zero_optimization = self.config.get('zero_optimization', {})
        return 'offload_optimizer' in zero_optimization

    def _set_raise_error_at_min_scale(self):
        optimizer = getattr(self.deepspeed_engine, 'optimizer', None)
        loss_scaler = getattr(optimizer, 'loss_scaler', None)
        if self.raise_error_at_min_scale is not None and loss_scaler is not None:
            loss_scaler.raise_error_at_min_scale = self.raise_error_at_min_scale

    def setup(self, trainer: L.Trainer) -> None:
        super().setup(trainer)

        self._set_raise_error_at_min_scale()
        self.model_to_device()

    def model_init_context(self):
        return super().model_sharded_context()

    @contextmanager
    def model_sharded_context(self):
        yield

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r = super().training_step(*args, **kwargs)
        self.lightning_module.trainer.progress_bar_metrics['skipped_steps'] = self.deepspeed_engine.skipped_steps
        return r

    def save_checkpoint(self, checkpoint: dict, filepath: _PATH, storage_options: Any | None = None) -> None:
        weights_only = 'optimizer_states' not in checkpoint

        filepath = self.broadcast(filepath)

        if storage_options is not None:
            raise TypeError(
                '`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg'
                f' is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used.'
            )
        
        _exclude_keys = ['state_dict', 'optimizer_states']
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag='checkpoint', exclude_frozen_parameters=self.exclude_frozen_parameters)

        def save_weights_only():
            checkpoint = get_lightning_checkpoint_from_zero_checkpoint(filepath)
            shutil.rmtree(filepath)
            checkpoint_io = self.checkpoint_io
            if isinstance(checkpoint_io, AsyncCheckpointIO):
                checkpoint_io = checkpoint_io.checkpoint_io
            checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options)

        if self.is_global_zero and weights_only:
            if isinstance(self.checkpoint_io, AsyncCheckpointIO):
                self.checkpoint_io._executor.submit(save_weights_only)
            else:
                save_weights_only()
