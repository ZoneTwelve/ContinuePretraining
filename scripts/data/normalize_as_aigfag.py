import fire


def map_fn(x):
    return (
        f"標題：{x['title']}\n"
        f"內文：{x['context']}\n"
        f"類別：{x['category']}\n"
        f"分組：{x['group']}\n"
        f"網址：{x['url']}\n"
        f"聯絡方式：{x['contact']}"
    )


def main(
    input_path: str,
    output_path: str
):
    import pandas as pd

    df = pd.read_csv(input_path)
    s = df.apply(map_fn, axis=1)
    df = s.to_frame('text')
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)
