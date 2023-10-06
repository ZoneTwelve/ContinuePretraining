import random

from .templater import Templater


class GeneralTemplater(Templater):    
    def match(self, type: str, **kwargs) -> bool:
        return type == 'general'
    
    def _add_instruction_end(self, instruction: str):
        if instruction[-1] in ['。', '？', '：']:
            return instruction + random.choice(['', ' ', '\n'])
        return instruction + random.choice(['\n', '：', '。', ' '])

    def apply(
        self,
        instruction: str,
        input: str,
        response: str,
        **kwargs
    ) -> tuple[str, str]:
        if not input:
            prompt = instruction
        else:
            prompt = self._add_instruction_end(instruction) + input

        return prompt, response
