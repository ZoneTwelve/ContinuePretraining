import os

import fire
import torch

from taide_cp.models import LitCausalLMConfig, LitCausalLM
from taide_cp.utils.deepspeed import \
    get_lightning_checkpoint_from_zero_checkpoint


_MODULE_MAPPING = {
    ('taide_cp.data.pre_training.pre_training_config', 'PreTrainingConfig'): ('taide_cp.data.pre_training.datamodule_for_pre_training_config', 'DataModuleForPreTrainingConfig'),
    ('taide_cp.data.pre_training.pre_training_config', 'ConcatMethod'): ('taide_cp.data.pre_training.datamodule_for_pre_training_config', 'ConcatMethod'),
}


def patch_unpickler():
    import pickle

    _Unpickler = pickle.Unpickler

    class Dummy:
        def __init__(self, *args, **kwargs) -> None:
            self.args, kwargs = args, kwargs

    class Unpickler(_Unpickler):
        def find_class(self, module_name: str, global_name: str):
            if (t := (module_name, global_name)) and t in _MODULE_MAPPING:
                module_name, global_name = _MODULE_MAPPING[t]

            try:
                return super().find_class(module_name, global_name)
            except:
                print('module_name', module_name)
                print('global_name', global_name)
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
    model = LitCausalLM(config)
    model.configure_model()
    model.load_state_dict(checkpoint['state_dict'], assign=True)
    model.model.generation_config.bos_token_id = model.tokenizer.bos_token_id
    model.model.generation_config.eos_token_id = model.tokenizer.eos_token_id
    model.model.generation_config.pad_token_id = model.tokenizer.pad_token_id
    model.model.save_pretrained(output_path)
    model.tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
