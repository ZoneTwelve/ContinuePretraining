import fire
import torch
from transformers import LlamaForCausalLM


def get_task_vectors(a: LlamaForCausalLM, b: LlamaForCausalLM):
    a = a.state_dict()
    b = b.state_dict()
    storage = {
        'task_vectors': {},
        'full_weights': {},
    }
    for k in a:
        storage['task_vectors'][k] = b[k] - a[k]
        if k in ['model.embed_tokens.weight', 'lm_head.weight']:
            storage['full_weights'][k] = b[k]
    return storage


def main(
    model1_path: str,
    model2_path: str,
    output_path: str
):
    kwargs = dict(
        torch_dtype='auto',
        low_cpu_mem_usage=True
    )
    m1 = LlamaForCausalLM.from_pretrained(model1_path, **kwargs)
    m2 = LlamaForCausalLM.from_pretrained(model2_path, **kwargs)
    tv = get_task_vectors(m1, m2)
    torch.save(tv, output_path)


if __name__ == '__main__':
    fire.Fire(main)
