import fire

from taide_cp.data import DataModuleForPreTraining
from taide_cp.models import AutoTokenizer


def main(
    data_path: str,
    tokenizer_path: str,
    dataset_path: str,
    sequence_length: int,
    num_proc: int | None = None,
):
    datamodule = DataModuleForPreTraining(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=None),
        sequence_length=sequence_length,
        data_path=data_path,
        dataset_path=dataset_path,
        num_proc=num_proc,
    )
    datamodule.prepare_data()


if __name__ == '__main__':
    fire.Fire(main)
