from typing import Optional

import lightning as L
from torch import Generator
from torch.utils.data import DataLoader

from .dummy_dataset import DummyDataset


class DummyDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        vocab_size: int,
        sequence_length: int = 2048,
        num_items: int = 1000,
        generator: Optional[Generator] = None,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_items = num_items
        self.generator = generator

    def _get_dataset(self):
        return DummyDataset(
            self.vocab_size,
            self.sequence_length,
            self.num_items,
            self.generator,
        )

    def _get_dataloader(self, *args, **kwargs):
        return DataLoader(
            self._get_dataset(),
            batch_size=self.batch_size,
            *args,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return self._get_dataloader(shuffle=True)
    
    def val_dataloader(self):
        return self._get_dataloader()
    
    def test_dataloader(self):
        return self._get_dataloader()
    
    def predict_dataloader(self):
        return self._get_dataloader()
