import os
from typing import Literal, Optional

import fire
from sentencepiece import SentencePieceTrainer


def main(
    data: str,
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: Literal['unigram', 'bpe', 'char', 'word'] = 'bpe',
    max_sentence_length: int = 4192,
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = False,
    num_threads: Optional[int] = None,
):
    num_threads = num_threads or os.cpu_count()
    SentencePieceTrainer.Train(
        input=data,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        num_threads=num_threads,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        normalization_rule_name='identity',
    )

if __name__ == '__main__':
    fire.Fire(main)
