from typing import List, Union

import fire
from tokenizers import (Tokenizer, decoders, normalizers, pre_tokenizers,
                        processors)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def main(
    files: Union[str, List[str]],
    output: str
):
    if isinstance(files, str):
        files = [files]

    trainer = BpeTrainer(
        show_progress=True,
    )
    bpe = BPE(
        unk_token='<unk>',
        fuse_unk=True,
        byte_fallback=True,
    )
    tokenizer = Tokenizer(bpe)
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(content=" ", left=1),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Punctuation(),
    ])
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Prepend(prepend="▁"),
        normalizers.Replace(pattern=" ", content="▁"),
    ])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        pair="<s> $A $B",
        special_tokens=[
            ("<s>", 1),
        ],
    )

    tokenizer.train(files, trainer)
    tokenizer.save(output)

if __name__ == '__main__':
    fire.Fire(main)
