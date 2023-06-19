import os
import re
from typing import *

if TYPE_CHECKING:
    from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.plugins import PLUGIN_INPUT
    from lightning.pytorch.strategies import Strategy
    from transformers import PreTrainedTokenizer

from ..utilities import parse_ev
from .decorators import component

__all__ = ['get_model_for_pre_training', 'get_strategy', 'get_datamodule_for_pre_training', 'get_wandb_logger', 'get_trainer']


@component()
def get_model_for_pre_training(
    model_type: str,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    extend_tokens: bool = False,
    initializing_strategy: Optional[str] = None,
    freezing_strategy: Optional[str] = None,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 1e-1,
    lr_scheduler_type: Optional[str] = None,
    num_warmup_steps: int = 0,
    min_lr_factor: float = 0.1,
    ckpt_path: Optional[str] = None,
):
    from taide_cp.training import MODELS_FOR_PRE_TRAINING
    
    model_cls = MODELS_FOR_PRE_TRAINING[model_type]
    return model_cls(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        extend_tokens=extend_tokens,
        initializing_strategy=initializing_strategy,
        freezing_strategy=freezing_strategy,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        min_lr_factor=min_lr_factor,
        _load_from_checkpoint=ckpt_path is not None,
    )


@component(ignored_keywords_on_entry_point=['tokenizer'])
def get_datamodule_for_pre_training(
    dataset_path: str,
    tokenizer: "PreTrainedTokenizer",
    micro_batch_size: int = 1,
    micro_batch_size_val: Optional[int] = None,
    num_workers: int = 4,
):
    from ...data import DataModuleForPreTraining
    
    return DataModuleForPreTraining(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=micro_batch_size,
        batch_size_val=micro_batch_size_val,
        num_workers=num_workers,
    )


@component()
def get_wandb_logger(
    name: Optional[str] = None,
    version: Optional[str] = None,
    save_dir: str = './logs/',
    project: str = 'taide_cp',
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
):
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        name=name,
        version=version,
        save_dir=save_dir,
        project=project,
        tags=re.split(r',\s*', tags) if tags else [],
        notes=notes,
        group=group,
        job_type=job_type,
    )


@component()
def get_strategy(strategy: str = 'auto', **kwargs) -> Union[str, "Strategy"]:
    if strategy == 'deepspeed':
        from ...lightning.deepspeed_strategy import DeepSpeedStrategy
        return DeepSpeedStrategy(**kwargs)
    return strategy


@component(ignored_keywords_on_entry_point=['strategy', 'logger', 'callbacks', 'plugins'])
def get_trainer(
    strategy: Union[str, "Strategy"] = 'auto',
    devices: Union[List[int], str, int] = 'auto',
    num_nodes: Optional[int] = None,
    precision: "_PRECISION_INPUT" = '16-mixed',
    logger: Optional[Union["Logger", Iterable["Logger"], bool]] = None,
    callbacks: Optional[Union[List["Callback"], "Callback"]] = None,
    max_epochs: Optional[int] = None,
    min_epochs: Optional[int] = None,
    max_steps: int = -1,
    min_steps: Optional[int] = None,
    val_check_interval: Optional[Union[int, float]] = None,
    check_val_every_n_epoch: Optional[int] = 1,
    enable_checkpointing: Optional[bool] = None,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: Optional[Union[int, float]] = None,
    plugins: Optional[Union["PLUGIN_INPUT", List["PLUGIN_INPUT"]]] = None,
):
    from lightning import Trainer
    from lightning.pytorch.plugins.environments import SLURMEnvironment

    num_nodes = num_nodes or parse_ev(int, 'SLURM_NNODES', 1)

    plugins = plugins or []
    if os.environ.get('SLURM_JOB_ID') is not None:
        plugins += [SLURMEnvironment(auto_requeue=False)]

    return Trainer(
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        min_epochs= min_epochs,
        max_steps=max_steps,
        min_steps=min_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=enable_checkpointing,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        plugins=plugins,
    )
