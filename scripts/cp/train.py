import logging

import fire
import multiprocess

from taide_cp.utils import SLURM
from taide_cp.utils.scripting import *


@entry_point(
    get_model_for_pre_training,
    get_datamodule_for_pre_training,
    get_logger,
    get_trainer,
)
def main(
    seed: int | None = None,
    ckpt_path: str | None = None,
    **kwargs,
):
    import lightning as L
    from lightning.pytorch.callbacks import (EarlyStopping,
                                             LearningRateMonitor,
                                             ModelCheckpoint)

    from taide_cp.lightning import (DeepSpeedSkippedStepsCallback,
                                    DeepSpeedStrategy,
                                    ResumeDataLoaderCallback)

    multiprocess.set_start_method('spawn')
    
    L.seed_everything(seed)

    model = get_model_for_pre_training(**kwargs)
    datamodule = get_datamodule_for_pre_training(
        tokenizer=model.tokenizer,
        sequence_length=model.max_length,
        **kwargs
    )

    strategy = DeepSpeedStrategy(
        stage=2,
        logging_batch_size_per_gpu=datamodule.batch_size['train'],
        logging_level=logging.ERROR,
    )

    callbacks = [
        ResumeDataLoaderCallback(),
        LearningRateMonitor(),
        EarlyStopping(monitor='Perplexity/Val'),
        ModelCheckpoint(
            save_on_train_epoch_end=True,
            save_top_k=-1
        ),
        ModelCheckpoint(
            save_top_k=1,
            save_weights_only=True,
            every_n_train_steps=500,
        ),
        ModelCheckpoint(
            monitor='Perplexity/Val',
            auto_insert_metric_name=False,
            filename='epoch={epoch}-step={step}-ppl={Perplexity/Val:.2f}',
        ),
    ]

    if isinstance(strategy, DeepSpeedStrategy):
        callbacks += [DeepSpeedSkippedStepsCallback()]

    logger = get_logger(**kwargs)
    logger.experiment.log_code('.')

    trainer = get_trainer(
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        **kwargs,
    )

    extra_hyperparameters_to_save = {
        'data_path': datamodule.data_path,
        'dataset_path': datamodule.dataset_path,
        'micro_batch_size': datamodule.batch_size['train'],
        'precision': trainer.precision,
        'accumulate_grad_batches': trainer.accumulate_grad_batches,
        'gradient_clip_val': trainer.gradient_clip_val,
        'seed': seed,
    }

    if SLURM.is_slurm:
        extra_hyperparameters_to_save['slurm'] = {
            'job_id': SLURM.job_id,
            'job_name': SLURM.job_name,
            'num_nodes': SLURM.num_nodes,
            'num_gpus': SLURM.num_tasks,
        }
    
    model.save_hyperparameters(extra_hyperparameters_to_save)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    fire.Fire(main)
