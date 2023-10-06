import json

from .templater import Templater


class PromptTemplater(Templater):
    template: dict[str, str]

    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.template = json.load(f)

    def apply(
        self,
        prompt: str,
        response: str,
        bos_token: str,
        eos_token: str,
        **kwargs
    ) -> str:
        prompt = self.template['prompt_template'].format(
            prompt=prompt,
            bos_token=bos_token
        )

        if kwargs.get('source') == 'alpaca_gpt4_tw' and response[-1] in ['。', '.']:
            eos_token = ''

        response = self.template['response_template'].format(
            response=response,
            eos_token=eos_token
        )
        return prompt, response
