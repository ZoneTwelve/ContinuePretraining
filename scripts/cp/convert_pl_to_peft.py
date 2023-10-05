import fire

from taide_cp.models import LightningModuleForPreTrainingWithLoRA


def _fix_import():
    import sys

    from taide_cp import models

    sys.modules['taide_cp.training'] = models


def main(
    checkpoint_path: str,
    output_path: str,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
):
    tokenizer, peft_model = LightningModuleForPreTrainingWithLoRA.convert_to_hf(
        checkpoint_path,
        model_path,
        tokenizer_path
    )
    # model.config.torch_dtype = 'float16'
    peft_model.save_pretrained(output_path, safe_serialization=True)
    # model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)

if __name__ == '__main__':
    fire.Fire(main)
