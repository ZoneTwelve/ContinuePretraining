import os
from typing import Any

from jsonargparse import Namespace
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.cloud_io import get_filesystem


class TaideCPSaveConfigCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace | Any,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
        disabled: bool = False
    ) -> None:
        super().__init__(parser, config, config_filename, overwrite, multifile, save_to_log_dir)

        self.already_saved = disabled

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
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
