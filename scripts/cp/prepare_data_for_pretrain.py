from typing import Optional, Union

import fire

from taide_cp.data import DataModuleForPreTraining
from taide_cp.models import AutoTokenizer


def main(
    data_path: str,
    tokenizer_path: str,
    dataset_path: str,
    sequence_length: int,
    num_proc: Optional[int] = None,
    val_split_size: Union[int, float] = 0.1,
):
    datamodule = DataModuleForPreTraining(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=None),
        sequence_length=sequence_length,
        data_path=data_path,
        dataset_path=dataset_path,
        num_proc=num_proc,
        val_split_size=val_split_size
    )
    datamodule.prepare_data()

if __name__ == '__main__':
    fire.Fire(main)
