import asyncio
import json
import re
import time
from typing import Any, Dict, List, Tuple

import aiohttp
from bs4 import BeautifulSoup
import fire
from tqdm.auto import tqdm

URL1 = 'https://dict.variants.moe.edu.tw/variants/rbt/standards_tiles.rbt?pageId=2982209'
URL2 = 'https://dict.variants.moe.edu.tw/variants/rbt/standards.rbt'


async def bootstrap(sessoin: aiohttp.ClientSession) -> Tuple[str, int]:
    response = await sessoin.get(URL1)
    text = await response.text()
    response.close()
    component_id = re.search(r'/variants/rbt/standards\.rbt\?componentId=([a-z0-9_]+)&rbtType=AJAX_INVOKE&pageCode=0', text).group(1)
    
    response = await sessoin.get(URL2,
        params={
            'componentId': component_id,
            'rbtType': 'AJAX_INVOKE',
        }
    )
    text = await response.text()
    response.close()

    soup = BeautifulSoup(text, features='html.parser')
    num_pages = int(soup.select_one('table:nth-child(1) span:nth-child(4)').text)
    return component_id, num_pages

def parse_page(text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(text, features='html.parser')
    trs = soup.select('table:nth-child(2) tr')
    header = [e.text for e in trs.pop(0).select('td')]

    items = []
    for tr in trs:
        column = [td.text.strip() for td in tr.select('td > :first-child')]
        x = {h: c for h, c in zip(header, column)}
        items.append(x)
    return items

async def scrape_page(semaphore: asyncio.Semaphore, session: aiohttp.ClientSession, component_id: str, page_code: int) -> List[Dict[str, Any]]:
    while True:
        async with semaphore:
            response = await session.get(URL2,
                params={
                    'componentId': component_id,
                    'rbtType': 'AJAX_INVOKE',
                    'pageCode': page_code,
                },
            )
            text = await response.text()
            response.close()
        
        if text:
            break
        
        await asyncio.sleep(1)
    
    items = parse_page(text)
    return items

async def main(
    output_path: str,
    num_workers: int = 1,
):
    tasks = []
    items = []
    session = aiohttp.ClientSession()
    component_id, num_pages = await bootstrap(session)
    progress = tqdm(total=num_pages, unit='page')
    semaphore = asyncio.Semaphore(num_workers)
    done_callback = lambda _: progress.update()
    for i in range(num_pages):
        task = asyncio.create_task(scrape_page(semaphore, session, component_id, i))
        task.add_done_callback(done_callback)
        tasks.append(task)

    await asyncio.gather(*tasks)
    await session.close()

    for task in tasks:
        items.extend(task.result())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    fire.Fire(main)
