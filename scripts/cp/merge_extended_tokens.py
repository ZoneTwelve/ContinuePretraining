import fire
import torch

from transformers import LlamaForCausalLM


def main(
    model_path: str,
    state_dict_path: str,
    output_path: str
):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.half, low_cpu_mem_usage=True)
    state_dict = torch.load(state_dict_path, 'cpu')

    embed_tokens_w2 = state_dict['_forward_module.model.model.embed_tokens.w2']
    lm_head_w2 = state_dict['_forward_module.model.lm_head.w2']

    num_new_tokens = embed_tokens_w2.size(0)
    n = model.config.vocab_size
    model.resize_token_embeddings(n + num_new_tokens)

    model.model.embed_tokens.weight.data[n:].copy_(embed_tokens_w2)
    model.lm_head.weight.data[n:].copy_(lm_head_w2)

    model.save_pretrained(output_path, max_shard_size='1024GB', safe_serialization=True)

if __name__ == '__main__':
    fire.Fire(main)
