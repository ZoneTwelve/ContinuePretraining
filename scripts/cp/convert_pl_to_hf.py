import os

import fire
import torch

from taide_cp.models import LitCausalLMConfig, LitLlamaForCausalLM
from taide_cp.utils.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


def main(
    checkpoint_path: str,
    output_path: str
):
    if os.path.isdir(checkpoint_path):
        checkpoint = convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path)

    config: LitCausalLMConfig = checkpoint['hyper_parameters']['config']
    model = LitLlamaForCausalLM(config)
    model.configure_sharded_model()
    model.load_state_dict(checkpoint['state_dict'])
    model.model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    model.tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
