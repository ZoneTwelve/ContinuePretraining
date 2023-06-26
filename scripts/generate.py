import functools
import json
from pathlib import Path
from typing import Any, Optional, cast

import fire
import torch
from accelerate import init_empty_weights
from transformers import GenerationConfig, PreTrainedModel

from taide_cp.models import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from taide_cp.utils import rsetattr


def load_checkpoint(model: PreTrainedModel, checkpoint_path: str, device: torch.device):
    state_dict = torch.load(checkpoint_path, device)
    for k in list(state_dict.keys()):
        state_dict[k.removeprefix('_forward_module.model.')] = state_dict.pop(k)
    
    for k, v in state_dict.items():
        rsetattr(model, k, torch.nn.Parameter(v))
    
    model.to(device)
    model.tie_weights()

def chunk(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = 'outputs/generation/',
    batch_size: int = 4,
    device: str = 'cuda'
):
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

    if 'pad_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'[PAD]': tokenizer.bos_token})

    if checkpoint_path is None:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map=device, torch_dtype='auto', low_cpu_mem_usage=True)
        model = cast(PreTrainedModel, model)
    else:
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        load_checkpoint(model, checkpoint_path, device)

    prompts = [
        'I believe the meaning of life is',
        'Simply put, the theory of relativity states that',
        'Building a website can be done in 10 simple steps:\n',
        'Tweet: "I hate it when my phone battery dies.\n"\nSentiment: Negative\n###\nTweet: "My day has been 👍"\nSentiment: Positive\n###\nTweet: "This is the link to the article"\nSentiment: Neutral\n###\nTweet: "This new music video was incredibile"\nSentiment:',
        'Translate English to French:\nsea otter => loutre de mer\npeppermint => menthe poivrée\nplush girafe => girafe peluche\ncheese =>',
        '我相信生命的意義在於',
        '簡單地說，相對論指出',
        '建立一個網站可以通過 10 個簡單的步驟完成：\n1.',
        '推文：“我討厭手機沒電了。“\n情緒：負面\n###\n推文：“我的一天是👍”\n情緒：正面\n###\n推文：“這是文章的鏈接”\n情緒：中性\n###\n推文：“這個新的音樂影片太棒了”\n情緒：',
        '將英文翻譯成中文：\nsea otter => 海獺\npeppermint => 薄荷\nplush girafe => 毛絨長頸鹿\ncheese =>',
        "幫我翻譯下列句子成中文： I am a Master student studying computer science, and my research focus is NLP.",
        "如果心情不好要怎麼辦\n你可以",
        "告訴我三種賺錢的方式\n1. ",
        "請解下列一元一次方程式: x + 3 = 10, x = ",
        "台灣最高的建築物是",
        "台灣最好的大學是",
        "I like chocolate = 我喜歡巧克力\nI wish you a nice day =",
        "以「我」為主角，寫出一個關於如何獲取有效學習策略的故事。\n我",
        "你從小就夢想成為一名作家，終於有一天你的夢想成真了。你寫下了令人驚艷的小說作品，但你的父母嚴厲看待你從事這條職業道路，認為你將來不穩定，且沒有前途。寫下你和父母間爭吵的情景。\n我"
    ]

    generation_config = GenerationConfig(
        max_new_tokens=32,
        num_beams=4,
        temperature=0.1,
        top_p=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.autocast('cuda', dtype=model.dtype):
        output_text = []
        for batch in chunk(prompts, batch_size):
            x = tokenizer(batch, return_tensors='pt', padding=True, return_token_type_ids=False).to(model.device)
            l = x['input_ids'].size(1)
            x = model.generate(**x, generation_config=generation_config)
            x = tokenizer.batch_decode(x[:, l:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_text += x

    for p, o in zip(prompts, output_text):
        print(p, end=' ||| ')
        print(o)
        print('=' * 100)

    if output_path:
        output_path: Path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / (Path(model_path).name + '.generation.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generation_config': generation_config.to_diff_dict(),
                'prompts': prompts,
                'outputs': output_text,
            }, f, ensure_ascii=False, indent=True)

if __name__ == '__main__':
    fire.Fire(main)
