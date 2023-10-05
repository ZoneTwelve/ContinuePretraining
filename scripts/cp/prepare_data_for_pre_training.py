import fire
import multiprocess


def main(
    data_path: str | list[str],
    tokenizer_path: str,
    dataset_path: str,
    max_length: int,
    num_proc: int | None = None,
):
    from taide_cp.data import DataModuleForPreTraining, DataModuleForPreTrainingConfig
    from taide_cp.models import AutoTokenizer

    multiprocess.set_start_method('spawn')

    datamodule = DataModuleForPreTraining(
        DataModuleForPreTrainingConfig(
            max_length=max_length,
            data_path=data_path,
            dataset_path=dataset_path,
            num_proc=num_proc,
        ),
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=None),
    )
    datamodule.prepare_data()
    num_tokens = datamodule.count_tokens()
    print(f'Total Tokens: {num_tokens}')


if __name__ == '__main__':
    fire.Fire(main)
