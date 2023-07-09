import glob
import os

from datasets import Dataset, concatenate_datasets, load_dataset

from ...utils import DatasetsContextManager, disable_output
from ..datamodule import LightningDataModuleX
from .datacollator_for_supervised_fine_tuning import \
    DataCollatorForSupervisedFineTuning


def normalize_dataset(dataset: Dataset):
    dataset = dataset.rename_columns({o: o.lower() for o in dataset.column_names})
    column_mapping = {'instruction': 'prompt', 'output': 'response'}
    column_mapping = {k: v for k, v in column_mapping.items() if k in dataset.column_names}
    dataset = dataset.rename_columns(column_mapping)
    if 'input' not in dataset.column_names:
        dataset = dataset.add_column('input', [None] * len(dataset))
    dataset = dataset.select_columns(['prompt', 'input', 'response'])
    return dataset


@DatasetsContextManager()
def load_datasets(data_path: str) -> Dataset:
    paths = []
    if os.path.isfile(data_path):
        paths = [data_path]
    else:
        paths = glob(os.path.join(data_path, '**/*.*'), recursive=True)

    datasets = []
    for p in paths:
        with disable_output():
            x = load_dataset('csv', data_files=p)['train']
        x = normalize_dataset(x)
        datasets.append(x)

    return concatenate_datasets(datasets)


class DataModuleForSupervisedFineTuning(LightningDataModuleX):
    default_split = 'test'
    datacollator_cls = DataCollatorForSupervisedFineTuning
    
    def prepare_data(self) -> None:
        dataset = load_datasets(self.data_path)
        dataset.save_to_disk(self.dataset_path)
