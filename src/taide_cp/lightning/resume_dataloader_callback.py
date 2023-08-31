from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter


class ResumableDataloader(DataLoader):
    @property
    def current_step(self):
        if not hasattr(self, '_current_step'):
            self.current_step = 0
        return self._current_step
    
    @current_step.setter
    def current_step(self, v: int):
        self._current_step = v

    def __iter__(self) -> _BaseDataLoaderIter:
        it = super().__iter__()
        for _ in range(self.current_step):
            next(it)
        return it


class ResumeDataloaderCallback(Callback):
    def _reload_dataloader(self, trainer: Trainer):
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]) -> None:
        trainer.fit_loop.load_state_dict(checkpoint['loops']['fit_loop'])
        self._reload_dataloader(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.fit_loop.epoch_loop.batch_progress.current.reset()
        self._reload_dataloader(trainer)
