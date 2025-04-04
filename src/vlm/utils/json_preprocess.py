import os
from pathlib import Path

import polars as pl


def preprocess_json(input_path: str, output_path: str):
    df = pl.read_json(input_path)

    image_root = "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/images"

    df = df.with_columns(
        pl.col("image").map_elements(lambda p: str(Path(image_root) / p), return_dtype=pl.Utf8)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.write_ndjson(output_path)


if __name__ == "__main__":
    preprocess_json(
        "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/images/metadata.jsonl",
    )
