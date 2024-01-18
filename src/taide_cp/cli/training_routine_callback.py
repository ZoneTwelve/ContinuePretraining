import os

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from taide_cp.utils.slurm import SLURM


class TrainingRoutineCallback(Callback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            def exclude_fn(path: str, root: str):
                from wandb.sdk.lib.filenames import exclude_wandb_fn
                excluded = any(os.path.relpath(path, root).startswith(x + os.sep) for x in ('.cache', 'logs'))
                return exclude_wandb_fn(path, root) or excluded
            logger.experiment.log_code(exclude_fn=exclude_fn)

        extra_hyperparameters_to_save = {
            'precision': trainer.precision,
            'accumulate_grad_batches': trainer.accumulate_grad_batches,
            'gradient_clip_val': trainer.gradient_clip_val,
            'seed': int(os.environ.get('PL_GLOBAL_SEED', -1)) ,
        }

        if SLURM.is_slurm:
            extra_hyperparameters_to_save['slurm'] = {
                'job_id': SLURM.job_id,
                'job_name': SLURM.job_name,
                'num_nodes': SLURM.num_nodes,
                'num_gpus': SLURM.num_tasks,
            }

        pl_module.save_hyperparameters(extra_hyperparameters_to_save)