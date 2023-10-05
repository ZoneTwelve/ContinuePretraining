import os
import random
import shutil
from tempfile import gettempdir
from types import MethodType
from typing import Literal, Mapping

import lightning as L
from datasets import (Dataset, DatasetDict, concatenate_datasets,
                      get_dataset_config_info, load_dataset, load_from_disk)
from torch.utils.data import DataLoader

from ..lightning import ResumableDataLoader
from ..utils import SLURM, DatasetsContextManager, cpu_count
from ..utils.config import ConfigBase
from .datacollator import DataCollator

StageType = Literal['fit', 'validate', 'test', 'predict']
SplitType = Literal['train', 'val', 'test', 'pred']

STAGE2SPLIT: Mapping[StageType, SplitType] = {
    'fit': 'train',
    'validate': 'val',
    'test': 'test',
    'predict': 'pred',
}


def get_random_dir_path(seed: int | None = None):
    s = 'abcdefghijklmnopqrstuvwxyz0123456789_'
    r = random.Random(seed)
    name = 'tmp' + ''.join(r.choice(s) for _ in range(8))
    return os.path.join(gettempdir(), name)


class DataModuleConfig(ConfigBase):
    data_path: str | list[str] | None = None
    dataset_path: str | None = None
    train_batch_size: int = 1
    val_batch_size: int | None = None
    test_batch_size: int | None = None
    pred_batch_size: int | None = None
    train_split_size: int | float | None = None
    val_split_size: int | float | None = None
    test_split_size: int | float | None = None
    pred_split_size: int | float | None = None
    split_seed: int | None = 42
    num_workers: int = 1
    pin_memory: bool = True
    num_proc: int | None = None

    def __post_init__(self):
        self.num_proc = self.num_proc or cpu_count()

        if isinstance(self.data_path, str):
            self.data_path = [self.data_path]

        if self.dataset_path is None:
            seed = SLURM.job_id or os.getpgid(os.getpid())
            self.dataset_path = get_random_dir_path(seed)


class LightningDataModuleX(L.LightningDataModule):
    default_split: SplitType = 'train'
    datacollator: DataCollator | None = None

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()

        self.save_hyperparameters(config.asdict())

        self.dataset = DatasetDict()
        self.config = config

        self.split_size: dict[SplitType, int | float | None] = {
            'train': None,
            'val': None,
            'test': None,
            'predict': None
        }
        self.set_split_size(
            config.train_split_size,
            config.val_split_size,
            config.test_split_size,
            config.pred_split_size
        )

        self.batch_size: dict[SplitType, int | None] = {
            'train': None,
            'val': None,
            'test': None,
            'predict': None
        }
        self.set_batch_size(
            config.train_batch_size,
            config.val_batch_size,
            config.test_batch_size,
            config.pred_batch_size
        )

        self.cleanup_dataset_path = config.dataset_path is None

    def set_batch_size(
        self,
        train: int | None = None,
        val: int | None = None,
        test: int | None = None,
        predict: int | None = None,
    ):
        self.batch_size['train'] = train or self.batch_size['train']
        self.batch_size['val'] = val or self.batch_size['val'] or train
        self.batch_size['test'] = test or self.batch_size['test'] or train
        self.batch_size['predict'] = predict or self.batch_size['predict'] or train

    def set_split_size(
        self,
        train: int | float | None = None,
        val: int | float | None = None,
        test: int | float | None = None,
        predict: int | float | None = None,
    ):            
        self.split_size['train'] = train
        self.split_size['val'] = val
        self.split_size['test'] = test
        self.split_size['predict'] = predict

    def _set_dataloader_method(self, split: str):
        method_name = f'{split}_dataloader'
        setattr(self, method_name, MethodType(getattr(self.__class__, method_name), self))

    def _unset_dataloader_method(self, split: str):
        method_name = f'{split}_dataloader'
        setattr(self, method_name, MethodType(getattr(L.LightningDataModule, method_name), self))


    @DatasetsContextManager()
    def split(self, dataset: Dataset):
        dataset_dict = DatasetDict()
        dataset = dataset.shuffle(seed=self.config.split_seed)

        size: dict[SplitType, int | float | None] = {}
        for sp, sz in self.split_size.items():
            if isinstance(sz, float):
                size[sp] = int(dataset.num_rows * sz)
            else:
                size[sp] = sz

        if (
            size['train'] is None
            and size['val'] is None
            and size['test'] is None
            and size['predict'] is None
        ):
            dataset_dict[self.default_split] = dataset
        elif (
            size['train'] is None
            and size['val'] is not None and size['val'] < dataset.num_rows
            and size['test'] is None
            and size['predict'] is None
        ):
            train_size = dataset.num_rows - size['val']
            dataset_dict['train'] = dataset.select(range(train_size))
            dataset_dict['val'] = dataset.select(range(train_size, dataset.num_rows))
        else:
            s = 0
            for sp, sz in size.items():
                if sz:
                    e = s + sz
                    dataset_dict[sp] = dataset.select(range(s, e))
                    s = e

        for split in ['train', 'val', 'test', 'predict']:
            if split in dataset_dict:
                self._set_dataloader_method(split)
            else:
                self._unset_dataloader_method(split)

        return dataset_dict
    
    def prepare_data(self) -> None:
        if self.is_dataset(self.config.dataset_path):
            return
        
        dataset: Dataset = concatenate_datasets([load_dataset(p)['train'] for p in self.config.data_path])
        dataset.save_to_disk(self.config.dataset_path)
    
    def setup(self, stage: StageType | None = None) -> None:
        self.dataset = load_from_disk(self.config.dataset_path)
        self.dataset = self.split(self.dataset)

    def train_dataloader(self):
        dataloader = ResumableDataLoader(
            self.dataset['train'],
            batch_size=self.batch_size['train'],
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            shuffle=True,
        )
        if self.trainer is not None:
            dataloader.current_step = self.trainer.fit_loop.batch_idx
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.dataset['val'],
            batch_size=self.batch_size['val'],
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            shuffle=False,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'],
            batch_size=self.batch_size['test'],
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.dataset['predict'],
            batch_size=self.batch_size['predict'],
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            shuffle=False,
        )
    
    def teardown(self, stage: StageType | None = None) -> None:
        if self.cleanup_dataset_path:
            shutil.rmtree(self.config.dataset_path, ignore_errors=True)

    @staticmethod
    def is_dataset(path: str):
        try:
            get_dataset_config_info(path)
            return True
        except FileNotFoundError:
            return False
