import os
from collections import Counter

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
        counter = Counter()
        dataset = dataset.select_columns(['source', 'length'])
        progress = tqdm(total=len(dataset), desc=f'Count tokens ({k})')
        for batch in dataset.iter(1000):
            batch_size = len(batch['length'])
            for source, length in zip(batch['source'], batch['length']):
                counter[source] += length
                counter['all'] += length
            tokens[k] = counter
            progress.set_postfix(tokens=counter['all'])
            progress.update(batch_size)
    return tokens


def main():
    cli = TaideCPLightningCLI(run=False)

    datamodule: DataModuleForPreTraining = cli.datamodule
    config = datamodule.config

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
