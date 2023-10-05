import glob
import json
import math
import os
import re

from tqdm.auto import tqdm

input_path = 'data/ancient_chinese/raw/20230905'
output_path = 'data/ancient_chinese/test.jsonl'

paths = glob.glob(os.path.join(input_path, '**/*.jsonl'), recursive=True)
pbar = tqdm(total=len(paths))

pattern = r'\u4E00-\u9FFF'
pattern += r'\u3400-\u4DBF'
pattern += r'\U00020000-\U0002A6DF'
pattern += r'\U0002A700-\U0002B73F'
pattern += r'\U0002B740-\U0002B81F'
pattern += r'\U0002B820-\U0002CEAF'
pattern += r'\U0002CEB0-\U0002EBEF'
pattern += r'\U00030000-\U0003134F'
pattern += r'\U00031350-\U000323AF'
pattern += r'\uF900-\uFAFF'
pattern += r'\U0002F800-\U0002FA1F'
pattern += r'\u3000-\u303F'
pattern += r'\uFF01-\uFF5E'
# pattern = rf'[^{pattern}]'

# print(pattern)

# chars = set()
try:
    with open(output_path, 'w', encoding='utf-8') as fo:
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            pbar.set_description(f'Loading `{name}`')
            with open(p, 'r', encoding='utf-8-sig') as f:
                text = ''
                for i, l in enumerate(f):
                    x = json.loads(l)
                    if isinstance(x['text'], float) and math.isnan(x['text']):
                        # pbar.write(f'{p}, {i + 1}, {l[:-1]}')
                        continue

                    if re.search(f'[{pattern}]', x['text']) is None:
                        continue

                    text += x['text'] + '\n'
                text = text[:-1]
                text = re.sub(r'[^\w\d\uFF01-\uFF5E\u3000-\u303F\s]', '', text)
                # chars.update(text)
            fo.write(json.dumps({'name': name, 'text': text}, ensure_ascii=False) + '\n')
            pbar.update()
except Exception as e:
    pbar.write(f'{p} {i + 1} {l[:-1]}')
    raise e

# excluded_characters = set()
# for c in chars:
#     if re.search(f'[^{pattern}]', c) is not None:
#         excluded_characters.add(c)

# print(excluded_characters)
# print(len(excluded_characters))
