import json
import tempfile
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import fire
from typing_extensions import Self

if TYPE_CHECKING:
    from sentencepiece import SentencePieceProcessor
    from transformers import (GemmaTokenizer, LlamaTokenizer,
                              PreTrainedTokenizer)


class Extender:
    extender_classes: list[Self] = []

    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer_path = tokenizer_path
        self.tokenizer = self.load_tokenizer(tokenizer_path)
    
    def __init_subclass__(cls) -> None:
        cls.extender_classes.append(cls)

    @classmethod
    def match(cls, tokenizer_config: dict[str, Any]) -> bool:
        return False
    
    @classmethod
    def get_extender(cls, tokenizer_path: str, **kwargs) -> Self:
        from transformers.models.auto.tokenization_auto import \
            get_tokenizer_config
        tokenizer_config = get_tokenizer_config(tokenizer_path)
        for extender_class in reversed(cls.extender_classes):
            if extender_class.match(tokenizer_config):
                return extender_class(tokenizer_path, **kwargs)
        return Extender(tokenizer_path, **kwargs)
    
    def load_tokenizer(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def extend(self, tokens: list[str]) -> int:
        return self.tokenizer.add_tokens(tokens)

    def add_pad_token(self, pad_token: str) -> int:
        self.pad_token = pad_token
        return self.tokenizer.add_special_tokens({'pad_token': pad_token})

    def get_extended_tokenizer(self) -> "PreTrainedTokenizer":
        return self.tokenizer


class SentencePieceExtender(Extender, ABC):
    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        super().__init__(tokenizer)

        self._get_spm_and_pieces()

    @classmethod
    def match(cls, tokenizer_config: dict[str, Any]) -> bool:
        return False

    @property
    @abstractmethod
    def sp_model(self) -> "SentencePieceProcessor": ...

    def _get_spm_and_pieces(self) -> None:
        from transformers.utils import sentencepiece_model_pb2_new as sp_model

        self.spm = sp_model.ModelProto()
        self.spm.ParseFromString(self.sp_model.serialized_model_proto())
        self.pieces = set(p.piece for p in self.spm.pieces)

    def _add_token(self, token: str) -> bool:
        from transformers.utils import sentencepiece_model_pb2_new as sp_model

        if token not in self.pieces:
            sp = sp_model.ModelProto().SentencePiece()
            sp.piece = token
            sp.score = 0
            self.spm.pieces.append(sp)
            self.pieces.add(token)
            return True
        
        return False

    def extend(self, tokens: list[str]) -> int:
        tokens = sorted(tokens, key=lambda x: (-len(x), x))
        n = 0
        for t in tokens:
            if self._add_token(t):
                n += 1
        return n
    
    def add_pad_token(self, pad_token: str) -> int:
        self.pad_token = pad_token
        return self._add_token(pad_token)

    @abstractmethod
    def get_extended_tokenizer(self) -> "PreTrainedTokenizer": ...


class LlamaExtender(SentencePieceExtender):
    tokenizer: "LlamaTokenizer"

    @classmethod
    def match(cls, tokenizer_config: dict[str, Any]) -> bool:
        return tokenizer_config['tokenizer_class'] == 'LlamaTokenizer'

    @property
    def sp_model(self) -> "SentencePieceProcessor":
        return self.tokenizer.sp_model
    
    def load_tokenizer(self, tokenizer_path: str) -> "LlamaTokenizer":
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(tokenizer_path)

    def get_extended_tokenizer(self) -> "LlamaTokenizer":
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
    

class GemmaExtender(SentencePieceExtender):
    tokenizer: "GemmaTokenizer"

    @classmethod
    def match(cls, tokenizer_config: dict[str, Any]) -> bool:
        return tokenizer_config['tokenizer_class'] == 'GemmaTokenizer'

    @property
    def sp_model(self) -> "SentencePieceProcessor":
        return self.tokenizer.sp_model

    def load_tokenizer(self, tokenizer_path: str) -> "GemmaTokenizer":
        from transformers import GemmaTokenizer
        return GemmaTokenizer.from_pretrained(tokenizer_path)

    def get_extended_tokenizer(self) -> "GemmaTokenizer":
        from transformers import GemmaTokenizer

        with tempfile.NamedTemporaryFile() as f:
            f.write(self.spm.SerializeToString())
            f.flush()

            tokenizer = GemmaTokenizer(f.name, **self.tokenizer.init_kwargs)
        
        return tokenizer


def load_vocab(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def main(
    tokenizer_path: str,
    output_path: str,
    vocab_path: str | None = None,
    pad_token: str | None = None,
):
    extender = Extender.get_extender(tokenizer_path)
    
    vocab = []
    if vocab_path is not None:
        vocab = load_vocab(vocab_path)

    if pad_token is not None:
        extender.add_pad_token(pad_token)

    extender.extend(vocab)
    tokenizer = extender.get_extended_tokenizer()
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(main)
