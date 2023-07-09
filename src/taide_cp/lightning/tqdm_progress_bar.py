from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm.auto import tqdm


class TQDMProgressBar(TQDMProgressBar):
    val_progress_bar: tqdm
    test_progress_bar: tqdm

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        metrics = super().get_metrics(trainer, pl_module)
        for k in list(metrics.keys()):
            metrics[k.removesuffix('_step')] = metrics.pop(k)
        return metrics
    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional["STEP_OUTPUT"],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.val_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional["STEP_OUTPUT"],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.test_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
