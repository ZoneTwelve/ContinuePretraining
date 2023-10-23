import fire
import multiprocess


def main(
    dataset_kwargs: dict,
    tokenizer_path: str,
    max_length: int,
    stride: int | None,
    concat_method: str,
    num_proc: int | None = None,
):
    from taide_cp.data import DataModuleForPreTraining, PreTrainingConfig
    from taide_cp.models import AutoTokenizer

    multiprocess.set_start_method('spawn')

    datamodule = DataModuleForPreTraining(
        PreTrainingConfig(
            AutoTokenizer.from_pretrained(tokenizer_path),
            dataset_kwargs=dataset_kwargs,
            max_length=max_length,
            stride=stride,
            concat_method=concat_method,
            num_proc=num_proc,
        ),
    )
    datamodule.prepare_data()


if __name__ == '__main__':
    fire.Fire(main)
