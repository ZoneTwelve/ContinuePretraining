import glob
import json

import fire
from tqdm.auto import tqdm


def main(
    input_path: str,
    output_path: str
):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for p in tqdm(glob.glob(input_path)):
            with open(p, 'r', encoding='utf-8') as f:
                for l in tqdm(f, desc=f'Processing {p}', leave=False):
                    x = json.loads(l)
                    output_file.write(x['text'])
                    output_file.write('\n')

if __name__ == '__main__':
    fire.Fire(main)
