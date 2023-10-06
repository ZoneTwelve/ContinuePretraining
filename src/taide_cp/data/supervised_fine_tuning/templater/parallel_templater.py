import json
import random
from string import Formatter

from .templater import Templater


class ParallelTemplater(Templater):
    templates: list[str]

    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
    
    def match(self, type: str, **kwargs) -> bool:
        return type == 'parallel'
    
    def _get_template_field_names(self, template: str):
        return [fname for _, fname, _, _ in Formatter().parse(template) if fname]

    def apply(self, text1: str, text2: str, lang1: str, lang2: str, **kwargs) -> tuple[str, str]:
        template = random.choice(self.templates)
        direction = random.choice(['12', '21'])
        
        template_kwargs = {}
        if direction == '12':
            template_kwargs['text'] = text1
            template_kwargs['lang1'] = lang1
            template_kwargs['lang2'] = lang2
            response = text2
        elif direction == '21':
            template_kwargs['text'] = text2
            template_kwargs['lang1'] = lang2
            template_kwargs['lang2'] = lang1
            response = text1

        if 'lang1' not in self._get_template_field_names(template):
            template_kwargs.pop('lang1')

        prompt = template.format(**template_kwargs)
        return prompt, response


class LZHParallelTemplater(ParallelTemplater):
    def match(self, type: str, **kwargs) -> bool:
        return type == 'parallel' and {'zh-Hant', 'lzh-Hant'} == {kwargs.get('lang1'), kwargs.get('lang2')}
    
    def _replace_lang_tag(self, lang: str):
        mapping = {
            'zh-Hant': ['白話文', '現代文'],
            'lzh-Hant': ['文言文', '古文'],
        }
        return random.choice(mapping[lang])
    
    def apply(self, text1: str, text2: str, lang1: str, lang2: str, **kwargs) -> str:
        lang1 = self._replace_lang_tag(lang1)
        lang2 = self._replace_lang_tag(lang2)
        return super().apply(text1, text2, lang1, lang2, **kwargs)
