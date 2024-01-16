from dataclasses import KW_ONLY

from taide_cp.utils.config import ConfigBase


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
