import os

import fire
import torch

from taide_cp.models import LitCausalLMConfig, LitLlamaForCausalLM
from taide_cp.utils.deepspeed import \
    get_lightning_checkpoint_from_zero_checkpoint


def patch_unpickler():
    import pickle

    _Unpickler = pickle.Unpickler

    class Dummy:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class Unpickler(_Unpickler):
        def find_class(self, __module_name: str, __global_name: str):
            try:
                return super().find_class(__module_name, __global_name)
            except:
                return Dummy

    pickle.Unpickler = Unpickler


def main(
    checkpoint_path: str,
    output_path: str,
    ignore_errors: bool = False
):
    if ignore_errors:
        patch_unpickler()
    
    if os.path.isdir(checkpoint_path):
        checkpoint = get_lightning_checkpoint_from_zero_checkpoint(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path)

    config: LitCausalLMConfig = checkpoint['hyper_parameters']['config']
    config.patchers.clear()
    model = LitLlamaForCausalLM(config)
    model.configure_model()
    checkpoint['dtype'] = torch.half
    model.to(checkpoint['dtype'])
    model.load_state_dict(checkpoint['state_dict'])
    model.model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    model.tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
