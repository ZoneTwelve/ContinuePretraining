import fire

from taide_cp.utils.scripting import *


@entry_point(
    get_logger
)
def main(
    model_path: str,
    data_path: str,
    max_length: int = 2048,
    batch_size: int = 4,
    num_datapoints: int | None = None,
    num_proc: int | None = None,
    **kwargs,
):
    from taide_cp.data import DataModuleForPreTraining
    from taide_cp.training import LightningModuleForPerplexity

    model = LightningModuleForPerplexity(
        model_path=model_path,
        max_length=max_length,
    )
    model.save_hyperparameters()
    
    tokenizer = model.tokenizer
    
    if 'pad_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': tokenizer.bos_token})

    datamodule = DataModuleForPreTraining(
        tokenizer=tokenizer,
        sequence_length=max_length,
        data_path=data_path,
        test_batch_size=batch_size,
        test_split_size=num_datapoints or 1.0,
        num_proc=num_proc,
    )

    trainer = get_trainer(
        logger=get_logger('csv', save_dir='logs/eval', **kwargs),
        enable_checkpointing=False,
    )
    outputs = trainer.test(model, datamodule)

    if trainer.is_global_zero:
        print(outputs)


if __name__ == '__main__':
    fire.Fire(main)
