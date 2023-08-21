import fire

from taide_cp.training import LightningModuleForPreTraining


def main(
    checkpoint_path: str,
    output_path: str,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
):
    tokenizer, model = LightningModuleForPreTraining.convert_to_hf(
        checkpoint_path,
        model_path,
        tokenizer_path
    )

    model.save_pretrained(output_path, max_shard_size='1000GB', safe_serialization=True)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
