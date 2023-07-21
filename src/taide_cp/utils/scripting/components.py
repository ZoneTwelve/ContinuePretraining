import inspect
import re
from typing import *

if TYPE_CHECKING:
    from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
    from lightning.fabric.utilities.types import _PATH
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.plugins import PLUGIN_INPUT
    from lightning.pytorch.strategies import Strategy
    from transformers import PreTrainedTokenizer

from ..slurm import SLURM
from .decorators import component

__all__ = [
    'get_model_for_pre_training',
    'get_strategy',
    'get_datamodule_for_pre_training',
    'get_logger',
    'get_trainer'
]


@component()
def get_model_for_pre_training(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    extend_tokens: bool = False,
    initializing_strategy: Optional[str] = None,
    freezing_strategy: Optional[str] = None,
    max_length: int = 2048,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 1e-1,
    lr_scheduler_type: Optional[str] = None,
    num_warmup_steps: int = 0,
    min_lr_factor: float = 0.1,
    ckpt_path: Optional[str] = None,
):
    from ...models import AutoConfig
    from ...training import MODELS_FOR_PRE_TRAINING
    
    config = AutoConfig.from_pretrained(model_path)
    for model_class in MODELS_FOR_PRE_TRAINING:
        if config.__class__ is model_class.config_class:
            break
    else:
        raise ValueError(f'Model type `{config.model_type}` is not supported.')
    
    return model_class(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        extend_tokens=extend_tokens,
        initializing_strategy=initializing_strategy,
        freezing_strategy=freezing_strategy,
        max_length=max_length,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        min_lr_factor=min_lr_factor,
        _load_from_checkpoint=ckpt_path is not None,
    )


@component(ignored_keywords_on_entry_point=['tokenizer', 'sequence_length'])
def get_datamodule_for_pre_training(
    tokenizer: "PreTrainedTokenizer",
    sequence_length: int,
    data_path: str,
    dataset_path: str,
    micro_batch_size: int = 1,
    micro_batch_size_val: int | None = None,
    val_split_size: int | float | None = 0.1,
    num_workers: int = 4,
):
    from ...data import DataModuleForPreTraining
    
    return DataModuleForPreTraining(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        data_path=data_path,
        dataset_path=dataset_path,
        train_batch_size=micro_batch_size,
        val_batch_size=micro_batch_size_val,
        val_split_size=val_split_size,
        num_workers=num_workers,
    )


@component()
def get_logger(
    logger_type: Literal['csv', 'wandb'] = 'wandb',
    save_dir: str = 'logs',
    name: str | None = None,
    version: str | None = None,
    project: str = 'taide_cp',
    tags: str | list[str] | None = None,
    notes: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
):
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    if logger_type == 'csv':
        name = inspect.signature(CSVLogger).parameters['name'].default if name is None else name
        return CSVLogger(
            save_dir=save_dir,
            name=name,
            version=version,
        )
    
    elif logger_type == 'wandb':
        tags = tags or []
        tags = re.split(r',\s*', tags) if isinstance(tags, str) else tags
        return WandbLogger(
            name=name,
            version=version,
            save_dir=save_dir,
            project=project,
            tags=tags,
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
    gradient_clip_val: Optional[Union[int, float]] = 1.0,
    plugins: Optional[Union["PLUGIN_INPUT", List["PLUGIN_INPUT"]]] = None,
    default_root_dir: Optional["_PATH"] = None,
):
    from lightning import Trainer
    from lightning.pytorch.callbacks import ProgressBar

    num_nodes = num_nodes or SLURM.num_nodes or 1
    
    callbacks = callbacks or []
    for x in callbacks:
        if isinstance(x, ProgressBar):
            break
    else:
        from ...lightning import TQDMProgressBar
        callbacks += [TQDMProgressBar()]

    plugins = plugins or []
    if SLURM.is_slurm:
        from lightning.pytorch.plugins.environments import SLURMEnvironment
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
        default_root_dir=default_root_dir
    )
