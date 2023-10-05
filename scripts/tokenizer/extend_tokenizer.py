import tempfile
from typing import TYPE_CHECKING

import fire

if TYPE_CHECKING:
    from transformers import LlamaTokenizer, PreTrainedTokenizer


class Extender:
    def __init__(self, tokenizer: "PreTrainedTokenizer", pad_token: str | None) -> None:
        self.tokenizer = tokenizer
        self.pad_token = pad_token

        if self.pad_token is not None:
            self.add_pad_token(pad_token)
    
    def extend(self, tokens: list[str]):
        self.tokenizer.add_tokens(tokens)

    def add_pad_token(self, pad_token: str):
        self.tokenizer.add_special_tokens({'pad_token': pad_token})

    def get_extended_tokenizer(self):
        return self.tokenizer


class LLaMAExtender(Extender):
    tokenizer: "LlamaTokenizer"

    def __init__(self, tokenizer: "LlamaTokenizer", pad_token: str | None) -> None:
        from transformers.utils import sentencepiece_model_pb2_new as sp_model
        self.spm = sp_model.ModelProto()
        self.spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
        self.pieces = set(p.piece for p in self.spm.pieces)

        super().__init__(tokenizer, pad_token)

    def _add_token(self, token: str):
        from transformers.utils import sentencepiece_model_pb2_new as sp_model

        if token not in self.pieces:
            sp = sp_model.ModelProto().SentencePiece()
            sp.piece = token
            sp.score = 0
            self.spm.pieces.append(sp)
            self.pieces.add(token)

    def extend(self, tokens: list[str]):
        for t in tokens:
            self._add_token(t)

    def add_pad_token(self, pad_token: str):            
        self._add_token(pad_token)

    def get_extended_tokenizer(self):
        from transformers import LlamaTokenizer

        with tempfile.NamedTemporaryFile() as f:
            f.write(self.spm.SerializeToString())
            f.flush()

            kwargs = {
                **self.tokenizer.init_kwargs,
                'pad_token': self.pad_token,
                'padding_side': 'right',
            }
            tokenizer = LlamaTokenizer(f.name, **kwargs)
        
        return tokenizer


def main(
    tokenizer_path: str,
    output_path: str,
    vocab_path: str | None = None,
    pad_token: str | None = '<pad>',
):
    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    from taide_cp.utils import read_json

    tokenizer_config = get_tokenizer_config(tokenizer_path)
    tokenizer_class = tokenizer_config.pop('tokenizer_class', None)
    tokenizer_kwargs = {}
    extender_class = Extender

    if tokenizer_class == 'LlamaTokenizer':
        tokenizer_kwargs['use_fast'] = False
        tokenizer_kwargs['legacy'] = False
        extender_class = LLaMAExtender

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    extender = extender_class(tokenizer, pad_token)
    vocabs = read_json(vocab_path) if vocab_path is not None else []
    extender.extend(vocabs)

    tokenizer = extender.get_extended_tokenizer()
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
