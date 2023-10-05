from dataclasses import field

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from ..datamodule import DataModuleConfig, LightningDataModuleX
from .datacollator_for_supervised_fine_tuning import \
    DataCollatorForSupervisedFineTuning
from .templater import Templater


class DataModuleForSupervisedFineTuningConfig(DataModuleConfig):
    max_length: int | None = None
    templaters: list[Templater] = field(default_factory=list)


class DataModuleForSupervisedFineTuning(LightningDataModuleX):
    def __init__(
        self,
        config: DataModuleForSupervisedFineTuningConfig,
        tokenizer: PreTrainedTokenizer
    ) -> None:
        super().__init__(config)

        self.tokenizer = tokenizer
        self.datacollator = DataCollatorForSupervisedFineTuning(tokenizer, config.templaters, config.max_length)

    def prepare_data(self) -> None:
        if self.is_dataset(self.config.dataset_path):
            return
        
        dataset: Dataset = concatenate_datasets([load_dataset('json', data_files=p)['train'] for p in self.config.data_path])
        dataset.save_to_disk(self.config.dataset_path)
