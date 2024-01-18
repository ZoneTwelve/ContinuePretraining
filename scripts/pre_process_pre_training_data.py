import os

import multiprocess
from datasets import DatasetDict
from tqdm.auto import tqdm

from taide_cp.cli import TaideCPLightningCLI
from taide_cp.data import *
from taide_cp.lightning import *
from taide_cp.models import *
from taide_cp.patchers import *


def count_tokens(dataset_dict: DatasetDict) -> dict[str, int]:
    tokens = {}
    for k, dataset in dataset_dict.items():
        tokens[k] = 0
        dataset = dataset.select_columns('length')
        progress = tqdm(total=len(dataset), desc=f'Count tokens ({k})')
        for batch in dataset.iter(1000):
            tokens[k] += sum(batch['length'])
            progress.update(len(batch['length']))
    return tokens


def main():
    multiprocess.set_start_method('spawn')

    cli = TaideCPLightningCLI(run=False)

    datamodule: DataModuleForPreTraining = cli.datamodule
    config = datamodule.config

    if config.num_proc > 56:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    if config.tokenizer.is_fast:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    datamodule = DataModuleForPreTraining(config)
    dataset_dict = datamodule.load_data()
    dataset_dict = datamodule.pre_process_data(dataset_dict)
    print(dataset_dict)
    dataset_dict.save_to_disk(config.dataset_path)

    original_tokens = count_tokens(dataset_dict)
    dataset_dict = datamodule.post_process_data(dataset_dict)
    print(dataset_dict)
    resampled_tokens = count_tokens(dataset_dict)

    print('Original Tokens', original_tokens)
    print('Resampled Tokens', resampled_tokens)


if __name__ == '__main__':
    main()
