import fire
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm.auto import tqdm

def set_task_vectors(
    model: LlamaForCausalLM,
    tv: dict[str, dict[str, torch.Tensor]],
    skip_embeddings: bool = False,
    coefficient: float = 1.0
):
    model_vocab_size = model.config.vocab_size
    tv_vocab_size = tv['full_weights']['model.embed_tokens.weight'].size(0)

    if tv_vocab_size > model_vocab_size:
        model.resize_token_embeddings(tv_vocab_size)

    for n, p in tqdm(list(model.named_parameters())):
        v = tv['task_vectors'][n].to(device=p.device) * coefficient
        
        if n in ['model.embed_tokens.weight', 'lm_head.weight']:
            if skip_embeddings:
                continue
            
            if tv_vocab_size < model_vocab_size:
                p.data[:tv_vocab_size] += v
                continue
                        
            p.data += v
        else:
            p.data += v
    return model


def main(
    model_path: str,
    task_vector_path: str | list[str],
    output_path: str,
    skip_embeddings: bool = False
):
    task_vector_paths = task_vector_path if isinstance(task_vector_path, list) else [task_vector_path]
    coefficients = [1 / len(task_vector_paths)] * len(task_vector_paths)

    kwargs = dict(
        torch_dtype='auto',
        low_cpu_mem_usage=True
    )
    model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
    
    for tv_path, c in zip(task_vector_paths, coefficients):
        tv = torch.load(tv_path)
        model = set_task_vectors(model, tv, skip_embeddings=skip_embeddings, coefficient=c)

    model.save_pretrained(output_path, safe_serialization=True, max_shard_size='1000GB')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
