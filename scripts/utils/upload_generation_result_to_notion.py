import json
from typing import Any, Dict, List, Optional, Union

import fire
from notion.block import (BulletedListBlock, CodeBlock, HeaderBlock, PageBlock,
                          SubheaderBlock)
from notion.client import NotionClient


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(
    token_v2: str,
    page: str,
    results: Union[List[str], str],
    captions: Optional[Union[List[str], str]] = None,
    prompt_color: str = 'blue',
    output_color: str = 'red',
):
    client = NotionClient(token_v2=token_v2)
    page: PageBlock = client.get_block(page)

    results = [results] if not isinstance(results, list) else results
    results = [read_json(p) for p in results]

    page.children.add_new(HeaderBlock, title='Generation Config')
    page.children.add_new(CodeBlock, title=results[0]['generation_config'], language='python')

    page.children.add_new(HeaderBlock, title='Generation Result')
    page.children.add_new(SubheaderBlock, title='Legend')
    page.children.add_new(BulletedListBlock, title='Prompt', color=prompt_color)
    page.children.add_new(BulletedListBlock, title='Output', color=output_color)

    for i in range(len(results[0]['prompts'])):
        for j, r in enumerate(results):
            c = captions[j] if captions else None
            p = r['prompts'][i]
            o = r['outputs'][i].removeprefix(p)

            block: CodeBlock = page.children.add_new(CodeBlock, language='Plain Text')
            block.set('properties.title', [[p, [['h', prompt_color]]], [o, [['b'], ['h', output_color]]]])
            block.set('properties.caption', [[c]])

if __name__ == '__main__':
    fire.Fire(main)
