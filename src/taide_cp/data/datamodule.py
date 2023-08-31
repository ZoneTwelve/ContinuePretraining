import os
import random
import shutil
from tempfile import gettempdir
from typing import Any, Dict, Literal, Mapping, Optional, Type, Union

import lightning as L
from datasets import (Dataset, DatasetDict, get_dataset_config_info,
                      load_dataset, load_from_disk)
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from ..lightning import ResumableDataLoader
from ..utils import SLURM, DatasetsContextManager, cpu_count
from .datacollator import DataCollator

StageType = Literal['fit', 'validate', 'test', 'predict']
SplitType = Literal['train', 'val', 'test', 'pred']


STAGE2SPLIT: Mapping[StageType, SplitType] = {
    'fit': 'train',
    'validate': 'val',
    'test': 'test',
    'predict': 'pred',
}


def get_random_dir_path(seed: Optional[int] = None):
    s = 'abcdefghijklmnopqrstuvwxyz0123456789_'
    r = random.Random(seed)
    name = 'tmp' + ''.join(r.choice(s) for _ in range(8))
    return os.path.join(gettempdir(), name)


class LightningDataModuleX(L.LightningDataModule):
    datacollator_cls: Optional[Type[DataCollator]] = None
    datacollator: Optional[DataCollator] = None
    default_split: SplitType = 'train'

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        train_batch_size: int = 1,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        pred_batch_size: Optional[int] = None,
        train_split_size: Optional[Union[int, float]] = None,
        val_split_size: Optional[Union[int, float]] = None,
        test_split_size: Optional[Union[int, float]] = None,
        pred_split_size: Optional[Union[int, float]] = None,
        split_seed: Optional[int] = 42,
        num_workers: int = 1,
        pin_memory: bool = True,
        num_proc: Optional[int] = None,
        datacollator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.dataset = DatasetDict()
        
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_proc = num_proc or cpu_count()

        self.split_size: Dict[SplitType, Union[None, int, float]] = {
            'train': None,
            'val': None,
            'test': None,
            'pred': None
        }
        self.set_split_size(train_split_size, val_split_size, test_split_size, pred_split_size)
        self.split_seed = split_seed

        self.batch_size: Dict[SplitType, int] = {
            'train': None,
            'val': None,
            'test': None,
            'pred': None
        }
        self.set_batch_size(train_batch_size, val_batch_size, test_batch_size, pred_batch_size)

        self.cleanup_dataset_path = False
        if dataset_path is None:
            self.cleanup_dataset_path = True

            seed = SLURM.job_id or os.getpgid(os.getpid())
            self.dataset_path = get_random_dir_path(seed)

        if self.datacollator_cls is not None:
            datacollator_kwargs = datacollator_kwargs or {}
            self.datacollator = self.datacollator_cls(tokenizer, **datacollator_kwargs)

    def set_batch_size(
        self,
        train: Optional[int] = None,
        val: Optional[int] = None,
        test: Optional[int] = None,
        pred: Optional[int] = None,
    ):
        self.batch_size['train'] = train or self.batch_size['train']
        self.batch_size['val'] = val or self.batch_size['val'] or train
        self.batch_size['test'] = test or self.batch_size['test'] or train
        self.batch_size['pred'] = pred or self.batch_size['pred'] or train

    def set_split_size(
        self,
        train: Optional[Union[int, float]] = None,
        val: Optional[Union[int, float]] = None,
        test: Optional[Union[int, float]] = None,
        pred: Optional[Union[int, float]] = None,
    ):            
        self.split_size['train'] = train
        self.split_size['val'] = val
        self.split_size['test'] = test
        self.split_size['pred'] = pred

    @DatasetsContextManager()
    def split(self, dataset: Dataset):
        dataset_dict = DatasetDict()
        dataset = dataset.shuffle(seed=self.split_seed)

        size: Dict[SplitType, Union[None, int, float]] = {}
        for sp, sz in self.split_size.items():
            if isinstance(sz, float):
                size[sp] = int(dataset.num_rows * sz)
            else:
                size[sp] = sz

        if (
            size['train'] is None
            and size['val'] is None
            and size['test'] is None
            and size['pred'] is None
        ):
            dataset_dict[self.default_split] = dataset
        elif (
            size['train'] is None
            and size['val'] is not None and size['val'] < dataset.num_rows
            and size['test'] is None
            and size['pred'] is None
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

        return dataset_dict
    
    def prepare_data(self) -> None:
        if self.is_dataset(self.dataset_path):
            return
        
        dataset = load_dataset(self.data_path)['train']
        dataset.save_to_disk(self.dataset_path)
    
    def setup(self, stage: Optional[StageType] = None) -> None:
        self.dataset = load_from_disk(self.dataset_path)
        self.dataset = self.split(self.dataset)

    def train_dataloader(self):
        if 'train' in self.dataset:
            dataloader = ResumableDataLoader(
                self.dataset['train'],
                batch_size=self.batch_size['train'],
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                collate_fn=self.datacollator,
                shuffle=True,
            )
            dataloader.current_step = self.trainer.fit_loop.batch_idx
            return dataloader

    def val_dataloader(self):
        if 'val' in self.dataset:
            return DataLoader(
                self.dataset['val'],
                batch_size=self.batch_size['val'],
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                collate_fn=self.datacollator,
                shuffle=False,
            )
        
    def test_dataloader(self):
        if 'test' in self.dataset:
            return DataLoader(
                self.dataset['test'],
                batch_size=self.batch_size['test'],
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                collate_fn=self.datacollator,
                shuffle=False,
            )
    
    def predict_dataloader(self):
        if 'predict' in self.dataset:
            return DataLoader(
                self.dataset['predict'],
                batch_size=self.batch_size['predict'],
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                collate_fn=self.datacollator,
                shuffle=False,
            )
    
    def teardown(self, stage: Optional[StageType] = None) -> None:
        if self.cleanup_dataset_path:
            shutil.rmtree(self.dataset_path, ignore_errors=True)

    @staticmethod
    def is_dataset(path: str):
        try:
            get_dataset_config_info(path)
            return True
        except FileNotFoundError:
            return False
