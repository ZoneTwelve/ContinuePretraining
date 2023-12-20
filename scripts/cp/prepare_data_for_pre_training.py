import logging

import fire
import multiprocess
from datasets import disable_caching

# disable_caching()

logging.getLogger('taide_cp').setLevel(logging.DEBUG)

def main(
    dataset_kwargs: dict,
    tokenizer_path: str,
    max_length: int,
    dataset_path: str,
    stride: int | None,
    concat_method: str,
    num_proc: int | None = None,
):
    from taide_cp.data import DataModuleForPreTraining, DataModuleForPreTrainingConfig
    from taide_cp.models import AutoTokenizer

    multiprocess.set_start_method('spawn')

    config = DataModuleForPreTrainingConfig(
        AutoTokenizer.from_pretrained(tokenizer_path),
        dataset_kwargs=dataset_kwargs,
        max_length=max_length,
        stride=stride,
        concat_method=concat_method,
        num_proc=num_proc,
    )
    
    datamodule = DataModuleForPreTraining(config)
    datamodule.setup()
    datamodule.save_to_disk(dataset_path)

    print(datamodule.dataset_dict)
    print('tokens:')
    print(datamodule.count_tokens())


if __name__ == '__main__':
    fire.Fire(main)
