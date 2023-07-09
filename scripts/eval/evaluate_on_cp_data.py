from typing import TYPE_CHECKING, Any, Dict, Optional

import fire

from taide_cp.utils.scripting import *

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


def get_unigram_probs(dataset: "Dataset", tokenizer: "PreTrainedTokenizer"):
    import torch
    from tqdm.auto import tqdm

    probs = torch.zeros(len(tokenizer))
    for x in tqdm(dataset):
        input_ids = torch.tensor(x['input_ids'])
        input_ids = input_ids[input_ids != -1]
        ids, counts = input_ids.unique(return_counts=True)
        probs[ids] += counts
    probs /= probs.sum()
    probs = probs.masked_fill(probs == 0, 1e-8)
    return probs


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

    # model.pplu.token_probs = torch.load(token_probs_path)
    # model.pplu.token_probs = get_unigram_probs(dataset, tokenizer)

    trainer = get_trainer(
        logger=get_logger('csv', save_dir='logs/eval', **kwargs),
        enable_checkpointing=False,
    )
    outputs = trainer.test(model, datamodule)

    if trainer.is_global_zero:
        print(outputs)


if __name__ == '__main__':
    fire.Fire(main)
