import os
import re
from typing import Literal, Optional, Tuple, Union, cast

import fire


def get_model(model_type, **kwargs):
    from taide_cp.training import (LightningModuleForPreTraining,
                                   LLaMALightningModuleForPreTraining,
                                   OPTLightningModuleForPreTraining)
    
    mapping = {
        'llama': LLaMALightningModuleForPreTraining,
        'opt': OPTLightningModuleForPreTraining,
    }
    model = mapping[model_type](**kwargs)
    model = cast(LightningModuleForPreTraining, model)
    return model

def main(
    model_type: Literal['llama', 'opt'],
    model_path: str,
    dataset_path: str,
    tokenizer_path: Optional[str] = None,
    project: str = 't-chatgpt',
    save_dir: str = 'logs',
    name: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    extend_tokens: bool = False,
    initializing_strategy: Optional[str] = None,
    freezing_strategy: Optional[str] = None,
    lr: float = 1e-5,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 1e-1,
    lr_scheduler_type: Optional[str] = None,
    num_warmup_steps: int = 0,
    min_lr_factor: float = 0.1,
    micro_batch_size: int = 1,
    micro_batch_size_val: Optional[int] = None,
    num_workers: int = 0,
    precision: str = '16-mixed',
    accumulate_grad_batches: int = 1,
    gradient_clip_val: int = 1.0,
    benchmark: bool = True,
    seed: Optional[int] = None,
    max_epochs: int = 1,
    max_steps: int = -1,
    val_check_interval: Optional[Union[int, float]] = None,
    ckpt_path: Optional[str] = None,
):
    import lightning as L
    from lightning.pytorch.callbacks import (LearningRateMonitor,
                                             ModelCheckpoint)
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.environments import SLURMEnvironment

    from taide_cp.data import PretrainDataModule
    from taide_cp.lightning import DeepSpeedStrategy

    L.seed_everything(seed)

    model = get_model(
        model_type,
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

    datamodule = PretrainDataModule(
        dataset_path=dataset_path,
        tokenizer=model.tokenizer,
        batch_size=micro_batch_size,
        batch_size_val=micro_batch_size_val,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            pin_memory=True,
            logging_batch_size_per_gpu=micro_batch_size,
        ),
        precision=precision,
        logger=[
            WandbLogger(
                project=project,
                save_dir=save_dir,
                name=name,
                version=version,
                tags=re.split(r',\s*', tags),
                notes=notes,
            )
        ],
        plugins=[SLURMEnvironment(auto_requeue=False)],
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor='epoch',
                mode='max',
                filename='e{epoch}',
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
            ),
            ModelCheckpoint(
                monitor='Perplexity/Val',
                auto_insert_metric_name=False,
                # filename='ppl={Perplexity/Val:.2f}',
                filename='s{step}',
            )
        ],
        max_epochs=max_epochs,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        benchmark=benchmark,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    extra_hyperparameters_to_save = {
        'micro_batch_size': micro_batch_size,
        'precision': precision,
        'accumulate_grad_batches': accumulate_grad_batches,
        'gradient_clip_val': gradient_clip_val,
        'seed': seed,
    }

    if 'SLURM_JOB_ID' in os.environ:
        extra_hyperparameters_to_save['slurm'] = {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'job_name': os.environ.get('SLURM_JOB_NAME'),
            'num_nodes': int(os.environ.get('SLURM_JOB_NUM_NODES')),
            'num_gpus': int(os.environ.get('SLURM_NTASKS'))
        }
    
    model.save_hyperparameters(extra_hyperparameters_to_save)

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == '__main__':
    fire.Fire(main)
