import tempfile

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.utils import sentencepiece_model_pb2_new as sp_model


def prune_tokenizer(tokenizer: LlamaTokenizer, ids_to_remove: list[int]):
    spm = sp_model.ModelProto()
    spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
    for i in sorted(ids_to_remove, reverse=True):
        del spm.pieces[i]
    
    with tempfile.NamedTemporaryFile('wb') as f:
        f.write(spm.SerializeToString())
        f.flush()
        tokenizer = LlamaTokenizer(f.name, **tokenizer.init_kwargs)

    return tokenizer


def prune_model(model: LlamaForCausalLM, ids_to_keep: list[int]):
    w1 = model.get_input_embeddings().weight.clone()
    w2 = model.get_output_embeddings().weight.clone()
    model.resize_token_embeddings(len(ids_to_keep))

    model.get_input_embeddings().weight.data = w1[ids_to_keep]
    model.get_output_embeddings().weight.data = w2[ids_to_keep]
    return model


def main(
    occurrence_path: str,
    model_path: str,
    output_path: str,
    min_occurrence: int = 1,
):
    occurrence = torch.load(occurrence_path)
    i2o: dict[int, int] = occurrence['i2o']
    ids_to_keep = [i for i, o in i2o.items() if i <= 32000 or o >= min_occurrence]
    ids_to_remove = [i for i, o in i2o.items() if i > 32000 and o < min_occurrence]

    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer = prune_tokenizer(tokenizer, ids_to_remove)

    model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        low_cpu_mem_usage=True
    )
    model = prune_model(ids_to_keep)
    
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size='1000GB'
    )

if __name__ == '__main__':
    main()
