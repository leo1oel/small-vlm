"""Prepare an N-sample overfit set for the text->image generation pathway.

Reads the first N records from the cached GPIC jsonl, downloads each image from
the Azure blob via AZURE_SAS_TOKEN, and writes a local samples.jsonl + images/
dir that devtools/gen_overfit_multi.py consumes. Token-safe: never prints the
SAS query string (the sig=...).

Run:
  .venv/bin/python devtools/gen_overfit_prep.py --n 100
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path

# Cluster CA bundle rejects Azure's chain; the SAS URL itself is trusted.
_SSL_CTX = ssl._create_unverified_context()

REPO = Path("/mmfs1/gscratch/krishna/leoym/small-vlm")
JSONL = Path.home() / ".cache" / "vlm" / "energon-jsonl" / "gpic" / "test" / "test.jsonl"
OUT = Path("/mmfs1/gscratch/krishna/leoym/gen_overfit_data_100")
CONTAINER = "data"
FOLDER = "gpic/test"


def _sas_url() -> str:
    env_path = REPO / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("AZURE_SAS_TOKEN="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    raise RuntimeError("AZURE_SAS_TOKEN not found in .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    sas = _sas_url()
    if "?" not in sas:
        raise RuntimeError("AZURE_SAS_TOKEN is not a full SAS URL (no '?')")
    endpoint, query = sas.split("?", 1)
    endpoint = endpoint.rstrip("/")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    records = []
    with JSONL.open() as f:
        for _ in range(args.n):
            records.append(json.loads(f.readline()))

    out_jsonl = []
    ok = 0
    for i, rec in enumerate(records):
        msgs = rec["messages"]
        img_path = msgs[0]["content"][0]["path"]  # images/<id>.jpg
        caption = msgs[1]["content"]
        if isinstance(caption, list):  # defensive: content may be a list of parts
            caption = " ".join(p.get("text", "") for p in caption if isinstance(p, dict))
        blob = f"{endpoint}/{CONTAINER}/{FOLDER}/{img_path}?{query}"
        dst = OUT / img_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(blob, timeout=30, context=_SSL_CTX) as r:  # noqa: S310
                dst.write_bytes(r.read())
            ok += 1
        except Exception as e:  # token-safe: print id + error class only, never the URL
            print(f"[prep] FAIL {rec['id'][:12]} {type(e).__name__}", flush=True)
            continue
        out_jsonl.append({"id": rec["id"], "image": img_path, "caption": caption,
                          "caption_type": rec.get("caption_type", "?")})
        if (i + 1) % 10 == 0:
            print(f"[prep] {i+1}/{len(records)} downloaded ({ok} ok)", flush=True)

    (OUT / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in out_jsonl))
    print(f"[prep] DONE: {ok}/{len(records)} images -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
