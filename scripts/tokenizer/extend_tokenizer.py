from typing import Optional

import fire


def main(
    tokenizer_path: str,
    output_path: str,
    vocab_path: Optional[str],
    pad_token: Optional[str] = None,
):
    from taide_cp.models import AutoTokenizer
    from taide_cp.utils import read_json

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocabs = read_json(vocab_path) if vocab_path is not None else []

    if pad_token is not None:
        tokenizer.add_special_tokens({'pad_token': pad_token})

    tokenizer.add_tokens(vocabs)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
