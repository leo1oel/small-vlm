import os
from pathlib import Path

import polars as pl

df = pl.read_json(
    "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
)

image_root = "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/images"

df = df.with_columns(
    pl.col("image").map_elements(lambda p: str(Path(image_root) / p), return_dtype=pl.Utf8)
)

output_path = "/pasteur2/u/yuhuiz/yiming/LLaVA/playground/data/LLaVA-Pretrain/images/metadata.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.write_ndjson(output_path)
