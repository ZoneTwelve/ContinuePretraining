import os

import multiprocess
from lightning import LightningModule, Trainer
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import (LightningArgumentParser, LightningCLI,
                                   SaveConfigCallback)
from lightning.pytorch.loggers import WandbLogger

from taide_cp.data import *
from taide_cp.lightning import *
from taide_cp.models import *
from taide_cp.patchers import *
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


class SaveConfigCallbackX(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        from lightning.fabric.utilities.cloud_io import get_filesystem

        logger = trainer.logger

        if self.already_saved:
            return

        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None

        if trainer.is_global_zero and isinstance(logger, WandbLogger):
            log_dir = os.path.join(log_dir, logger.name, logger.version)
        log_dir = trainer.strategy.broadcast(log_dir)

        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True

            if isinstance(logger, WandbLogger):
                logger.experiment.save(config_path, policy='now')

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)


class CustomLightningCLI(LightningCLI):  
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments('data.init_args.config.batch_size', 'trainer.strategy.init_args.logging_batch_size_per_gpu')
        parser.link_arguments('model.tokenizer', 'data.init_args.config.tokenizer', apply_on='instantiate')

        parser.add_lightning_class_args(TrainingRoutineCallback, 'training_routine_callback')

    def before_instantiate_classes(self) -> None:
        config = self.config.get(self.config.subcommand)
        extra_plugins = []
        if SLURM.is_slurm and SLURM.num_tasks > 1:
            extra_plugins += [{'class_path': 'SLURMEnvironment', 'init_args': {'auto_requeue': False}}]
        else:
            extra_plugins += [{'class_path': 'LightningEnvironment'}]
        config.trainer.plugins = config.trainer.plugins or [] + extra_plugins

if __name__ == '__main__':
    multiprocess.set_start_method('spawn')
    CustomLightningCLI(
        save_config_callback=SaveConfigCallbackX,
        save_config_kwargs={'overwrite': True},
        parser_kwargs={'parser_mode': 'omegaconf'}
    )
