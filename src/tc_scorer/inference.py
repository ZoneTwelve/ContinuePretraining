from typing import Any, List
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
import torch
from tqdm import tqdm

from .model import BertStyleScorer


class TwCnStyleEvaluator:
    def __init__(self, pretrained_path: str,  device: str='cuda') -> None:
        self.device = device
        cfg = BertConfig.from_pretrained(pretrained_path)
        if cfg.architectures[0] == 'BertStyleScorer':
            pretrained_model = BertStyleScorer
        elif cfg.architectures[0] == 'BertForSequenceClassification':
            pretrained_model = BertForSequenceClassification
        else:
            raise ValueError('Unknown model type')
        
        self.model = pretrained_model.from_pretrained(pretrained_path)
        self.model.eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    
    @torch.no_grad()
    def __call__(self, list_of_text: List[str], batch_size: int=64, pbar: bool=False) -> torch.FloatTensor:
        """
        :param list_of_text: list of text to be evaluated
        :param batch_size: batch size for inference
        :param pbar: whether to show progress bar
        return score: torch.FloatTensor of size (len(list_of_text)), 1 is closer to Taiwan style, 0 is closer to China style
        """
        tot_scores = []
        
        for i in tqdm(range(0, len(list_of_text), batch_size), disable=not pbar): # for each batch
            data = list_of_text[i:i+batch_size]
            inputs = self.tokenize(data)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            if outputs.logits.shape[1] == 1:
                logits = torch.sigmoid(outputs.logits).squeeze(-1).detach().cpu()
            else:
                logits = torch.softmax(outputs.logits, dim=-1).detach().cpu()[...: 1].squeeze(-1)
            tot_scores.append(logits)
            
        return torch.cat(tot_scores, dim=0)
        
    def tokenize(self, list_of_text: List[str]) -> Any:
        return self.tokenizer(list_of_text, 
                              padding=True, 
                              truncation=True, 
                              return_token_type_ids=False, 
                              return_tensors='pt')


if __name__ == '__main__':
    import json
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/best')
    parser.add_argument('--input_path', type=str, default='./data/test.jsonl')
    parser.add_argument('--col_name', type=str, default='text')
    parser.add_argument('--output_path', type=str, default='./data/test_score.jsonl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # load jsonl file
    with open(args.input_path, 'r') as f:
        data = [json.loads(line)[args.col_name] for line in f]
    
    # load model
    evaluator = TwCnStyleEvaluator(args.pretrained_path, device=args.device)
    score = evaluator(data, batch_size=args.batch_size, pbar=True)
    
    # write to jsonl file with two columns: text, score
    with open(args.output_path, 'w') as f:
        for text, score in zip(data, score):
            f.write(json.dumps({'text': text, 'score': score.item()}, ensure_ascii=False) + '\n')
    
    # statistics
    print('mean:', score.mean().item())
