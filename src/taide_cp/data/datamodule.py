from dataclasses import KW_ONLY
from functools import partial
from typing import Any, Callable

import lightning as L
from datasets import Dataset, DatasetDict, Features, load_dataset
from datasets.fingerprint import (Hasher, format_kwargs_for_fingerprint,
                                  format_transform_for_fingerprint,
                                  update_fingerprint)
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ..utils.config import ConfigBase
from .datacollator import DataCollator


class DataModuleConfig(ConfigBase):
    _: KW_ONLY
    dataset_kwargs: dict
    batch_size: int = 1
    num_proc: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    cleanup_cache_files: bool = False
    prepare_data_per_node: bool = False


class DataModule(L.LightningDataModule):
    datacollator: DataCollator | None = None

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()

        self.save_hyperparameters({'datamodule_config': config.get_config_to_log()})

        self.config = config
        self.prepare_data_per_node = config.prepare_data_per_node

    def _prepare_data(self, current_hook: str | None = None) -> DatasetDict:
        dataset_dict = load_dataset(**self.config.dataset_kwargs)

        if current_hook == 'prepare_data' and self.config.cleanup_cache_files:
            dataset_dict.cleanup_cache_files()

        split = self.config.dataset_kwargs.get('split', None)
        if split:
            dataset_dict = DatasetDict({split: dataset_dict})
        return dataset_dict

    def prepare_data(self) -> None:
        self._prepare_data('prepare_data')

    def _get_dataloader(self, split: str, shuffle: bool | None = None):
        return DataLoader(
            self.dataset_dict[split],
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            pin_memory=self.config.pin_memory
        )

    def setup(self, stage: str | None = None) -> None:        
        self.dataset_dict = self._prepare_data('setup')

        mapping = {
            'train': 'train_dataloader',
            'validation': 'val_dataloader',
            'test': 'test_dataloader'
        }

        for k, v in mapping.items():
            if k in self.dataset_dict:
                setattr(self, v, partial(self._get_dataloader, k, shuffle=k == 'train'))
            else:
                setattr(self, v, getattr(super(), v))

    def train_dataloader(self): ...

    def val_dataloader(self): ...
    
    def test_dataloader(self): ...

    @classmethod
    def hash_fn_kwargs(cls, fn_kwargs: dict[str, Any]):
        x = {}
        for k, v in fn_kwargs.items():
            if isinstance(v, PreTrainedTokenizerBase):
                s: dict = v.__getstate__()
                s.pop('tokens_trie', None)
                x[k] = Hasher.hash(s)
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
        desc: str | None = None,
        cache_file_name_fn: Callable[[str], str] | None = None
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
            'fn_kwargs': cls.hash_fn_kwargs(fn_kwargs),
        }

        if new_fingerprint is None:
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint['fingerprint_name'] = 'new_fingerprint'
            new_fingerprint = update_fingerprint(dataset._fingerprint, transform, kwargs_for_fingerprint)
        
        if cache_file_name_fn is not None:
            cache_file_name = cache_file_name_fn(new_fingerprint)

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
        remove_columns: str | list[str] | None = None,
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
                remove_columns=remove_columns,
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
