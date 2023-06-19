import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import fire
import torch
from datasets import (Dataset, concatenate_datasets, disable_caching,
                      load_dataset)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerFast)

from taide_cp.data import DataCollatorForPreTraining
from taide_cp.metrics import Perplexity


def load_datasets(data_dir: Union[str, Path]) -> Dataset:
    data_dir = Path(data_dir)
    datasets = []
    for p in data_dir.glob('*'):
        x = load_dataset(
            'json',
            data_files=str(p),
        )['train']
        datasets.append(x)
    return concatenate_datasets(datasets)

def tokenize(batch: List[str], tokenizer: PreTrainedTokenizerFast):
    batch = tokenizer(batch, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
    for y in batch['input_ids']:
        y[:] = [
            tokenizer.bos_token_id,
            *y,
            tokenizer.eos_token_id
        ]
    return batch

def rearrange_datapoints(
    batch: Dict[str, List[int]],
    max_length: int,
):
    datapoints = []
    input_ids = []
    for x in batch['input_ids']:
        input_ids += x
        while len(input_ids) >= max_length:
            datapoint = input_ids[:max_length]
            datapoints.append(datapoint)
            input_ids[:max_length] = []

    if input_ids:
        num_paddings = max_length - len(input_ids)
        datapoint = [-1] * num_paddings + input_ids
        datapoints.append(datapoint)
    
    batch['input_ids'] = datapoints
    return batch

def main(
    model_path: str,
    dataset_path: str,
    batch_size: int = 4,
):
    disable_caching()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': tokenizer.bos_token})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.half,
        device_map='auto'
    )
    model = cast(PreTrainedModel, model)
    # model = model.cuda()

    dataset = load_datasets(dataset_path)
    dataset = dataset.map(
        tokenize,
        fn_kwargs=dict(tokenizer=tokenizer),
        input_columns='text',
        remove_columns=dataset.column_names,
        batched=True,
    )

    dataset = dataset.map(
        rearrange_datapoints,
        fn_kwargs=dict(
            max_length=2048,
        ),
        batched=True,
        num_proc=4,
    )
    dataset = dataset.select(range(1000))
        
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=DataCollatorForPreTraining(tokenizer))
    perplexity = Perplexity(ignore_index=-100).to(model.device)

    pbar = tqdm(total=dataset.num_rows)
    with torch.inference_mode():
        for batch in dataloader:
            batch['input_ids'] = batch['input_ids'].to(model.device)
            batch['attention_mask'] = batch['attention_mask'].to(model.device)
            batch['labels'] = batch['labels'].to(model.device)
            x = model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            perplexity.update(x.loss, batch['labels'], by_loss=True)
            pbar.update(batch_size)
            pbar.set_postfix(ppl=perplexity.compute().cpu().item())

    ppl = perplexity.compute().cpu().item()
    loss = math.log(ppl)
    pbar.write(f'loss: {loss:.2f}')
    pbar.write(f'ppl: {ppl:.2f}')

if __name__ == '__main__':
    fire.Fire(main)
