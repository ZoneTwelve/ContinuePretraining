import logging
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Callable

import lightning as L
from datasets import (Dataset, DatasetDict, Features, load_dataset,
                      load_from_disk)
from datasets.fingerprint import (Hasher, format_kwargs_for_fingerprint,
                                  format_transform_for_fingerprint,
                                  update_fingerprint)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from transformers import PreTrainedTokenizerBase

from ..utils.config import ConfigBase
from .datacollator import DataCollator

logger = logging.getLogger(__name__)

class ResumableDataLoader(DataLoader):
    
    def __init__(
        self,
        datamodule: "DataModule",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._datamodule = datamodule

    def __iter__(self) -> _BaseDataLoaderIter:
        current_step = 0
        if self._datamodule._resume is not None:
            if self._datamodule.trainer.current_epoch == self._datamodule._resume['current_epoch']:
                current_step = self._datamodule._resume['current_step']
        return (x for i, x in enumerate(super().__iter__()) if i >= current_step)


class DataModuleConfig(ConfigBase):
    _: KW_ONLY
    dataset_kwargs: dict | None = None
    validation_split: int | float | None = None
    dataset_path: str | None = None
    batch_size: int = 1
    num_proc: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    cleanup_cache_files: bool = False
    prepare_data_per_node: bool = False

    def __post_init__(self):
        assert self.dataset_kwargs is not None or self.dataset_path is not None

class DataModule(L.LightningDataModule):
    datacollator: DataCollator | None = None

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()

        self.save_hyperparameters({'datamodule_config': config})

        self.config = config
        self.prepare_data_per_node = config.prepare_data_per_node
        
        self._resume = None

    def load_data(self) -> DatasetDict:
        assert self.config.dataset_kwargs is not None
        
        dataset_dict = load_dataset(**self.config.dataset_kwargs)

        if (split := self.config.dataset_kwargs.get('split', False)):
            dataset_dict = DatasetDict({split: dataset_dict})

        assert self.config.validation_split is None or 'train' in dataset_dict and len(dataset_dict) == 1

        return dataset_dict

    def process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        return dataset_dict
        
    def prepare_data(self) -> None:
        if self.config.dataset_path is None:
            dataset_dict = self.load_data()
            if self.config.cleanup_cache_files:
                dataset_dict.cleanup_cache_files()
            self.process_data(dataset_dict)
    
    def _get_dataloader(self, split: str):
        dataloader_class = DataLoader
        dataloader_kwargs = dict(
            dataset=self.dataset_dict[split],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            pin_memory=self.config.pin_memory
        )

        if split == 'train':
            dataloader_class = ResumableDataLoader
            dataloader_kwargs['shuffle'] = True
            dataloader_kwargs['datamodule'] = self

        return dataloader_class(**dataloader_kwargs)
    
    def _maybe_split_validation_set(self, dataset_dict: DatasetDict):
        if self.config.validation_split is not None:
            dataset_dict = dataset_dict['train'].train_test_split(self.config.validation_split, seed=42)
            dataset_dict['validation'] = dataset_dict.pop('test')
        return dataset_dict

    def setup(self, stage: str | None = None) -> None:
        if self.config.dataset_path is None:
            dataset_dict = self.load_data()
            dataset_dict = self.process_data(dataset_dict)
        else:
            logger.info('Load processed data from disk')
            dataset_dict = load_from_disk(self.config.dataset_path)
            logger.info('Done')

        self.dataset_dict = self._maybe_split_validation_set(dataset_dict)

        mapping = {
            'train': 'train_dataloader',
            'validation': 'val_dataloader',
            'test': 'test_dataloader',
            'predict': 'predict_dataloader'
        }

        for k, v in mapping.items():
            if k in self.dataset_dict:
                setattr(self, v, partial(self._get_dataloader, k))
            else:
                setattr(self, v, getattr(super(), v))

    def save_to_disk(self, dataset_path: str):
        assert self.dataset_dict is not None
        self.dataset_dict.save_to_disk(dataset_path, num_proc=self.config.num_proc)

    def train_dataloader(self): ...

    def val_dataloader(self): ...
    
    def test_dataloader(self): ...

    def predict_dataloader(self): ...

    def state_dict(self) -> dict[str, Any]:
        return {
            '_resume': {
                'current_epoch': self.trainer.current_epoch,
                'current_step': self.trainer.fit_loop.batch_idx
            }
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._resume = state_dict['_resume']

    @classmethod
    def hash_fn_kwargs(cls, fn_kwargs: dict[str, Any]):
        x = {}
        for k, v in fn_kwargs.items():
            if isinstance(v, PreTrainedTokenizerBase):
                x[k] = Hasher.hash(v.init_kwargs)
            else:
                x[k] = v
        return Hasher.hash(x)
    
    @classmethod
    def map_dataset(
        cls,
        dataset: Dataset,
        function: Callable | None = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: str | list[str] | None = None,
        batched: bool = False,
        batch_size: int | None = 1000,
        drop_last_batch: bool = False,
        remove_columns: str | list[str] | None = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool | None = None,
        cache_file_name: str | None = None,
        writer_batch_size: int | None = 1000,
        features: Features | None = None,
        disable_nullable: bool = False,
        fn_kwargs: dict | None = None,
        num_proc: int | None = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: str | None = None,
        desc: str | None = None
    ):
        dataset_kwargs = {
            'shard': dataset,
            'function': function,
            'with_indices': with_indices,
            'with_rank': with_rank,
            'input_columns': input_columns,
            'batched': batched,
            'batch_size': batch_size,
            'drop_last_batch': drop_last_batch,
            'remove_columns': remove_columns,
            'keep_in_memory': keep_in_memory,
            'writer_batch_size': writer_batch_size,
            'features': features,
            'disable_nullable': disable_nullable,
            'fn_kwargs': cls.hash_fn_kwargs(fn_kwargs) if fn_kwargs is not None else fn_kwargs
        }

        if new_fingerprint is None:
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint['fingerprint_name'] = 'new_fingerprint'
            new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)

        dataset = dataset.map(
            function=function,
            with_indices=with_indices,
            with_rank=with_rank,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
            desc=desc,
        )
        
        return dataset

    @classmethod
    def map_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        function: Callable | None = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: str | list[str] | None = None,
        batched: bool = False,
        batch_size: int | None = 1000,
        drop_last_batch: bool = False,
        remove_columns: str | list[str] | bool | None = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool | None = None,
        cache_file_names: dict[str, str | None] | None = None,
        writer_batch_size: int | None = 1000,
        features: Features | None = None,
        disable_nullable: bool = False,
        fn_kwargs: dict | None = None,
        num_proc: int | None = None,
        desc: str | None = None,
    ):
        if cache_file_names is None:
            cache_file_names = {k: None for k in dataset_dict}

        return DatasetDict({
            k: cls.map_dataset(
                dataset,
                function=function,
                with_indices=with_indices,
                with_rank=with_rank,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                remove_columns=dataset.column_names if remove_columns == True else remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[k],
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc=desc
            ) for k, dataset in dataset_dict.items()
        })
