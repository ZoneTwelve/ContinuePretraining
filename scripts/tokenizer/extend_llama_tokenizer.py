import tempfile
from typing import Optional

import fire
from sentencepiece import sentencepiece_model_pb2 as sp_model
from transformers import LlamaTokenizer

from taide_cp.utils import read_json


def add_token(spm, token: str):
    sp = sp_model.ModelProto().SentencePiece()
    sp.piece = token
    sp.score = 0
    spm.pieces.append(sp)

def main(
    tokenizer_path: str,
    output_path: str,
    vocab_path: Optional[str] = None,
    pad_token: Optional[str] = '<pad>',
):
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    vocabs = read_json(vocab_path) if vocab_path is not None else []

    spm = sp_model.ModelProto()
    spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

    if pad_token is not None:
        add_token(spm, pad_token)

    token_set = set(p.piece for p in spm.pieces)
    for v in vocabs:
        if v not in token_set:
            add_token(spm, v)

    with tempfile.NamedTemporaryFile() as f:
        f.write(spm.SerializeToString())
        f.flush()

        kwargs = {
            **tokenizer.init_kwargs,
            'pad_token': '<pad>',
            'padding_side': 'left',
        }
        tokenizer = LlamaTokenizer(f.name, **kwargs)
    
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    fire.Fire(main)
