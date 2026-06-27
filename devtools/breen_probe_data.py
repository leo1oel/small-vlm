#!/usr/bin/env python
"""Non-hanging existence probe for the streamed BREEN caption data.

The captain's `train.jsonl` lands LAST and is the upload-complete signal
(training-directive.md). This checks for it via an MSC metadata HEAD
(`client.info`).

VALIDATED behavior (2026-06-24): info() returns FAST on an EXISTING blob (size
only, independent of file size) but HANGS on a MISSING one — MSC retries the
Azure 404 indefinitely (same root cause as the `client.open` hang). So the
caller MUST wrap this in a hard `timeout`; a missing blob surfaces as the
caller's timeout (exit 124), not a clean exit 1. The orchestrator's contract is
therefore simply: exit 0 == ready, ANY non-zero (1/2/124) == keep waiting.

Exit codes (consumed by breen_orchestrate.sh):
  0  -> train.jsonl exists, non-empty (prints "FOUND <bytes>")
  1  -> definitively absent, fast 404 (prints "MISSING") [rare — usually hangs]
  2  -> probe error                   (prints "ERROR <msg>")

Usage:
  python devtools/breen_probe_data.py msc://azure/data/yiming/bee_stage2/train/train.jsonl
"""

import sys

# Importing the project's energon module runs _bootstrap_env() at import time,
# deriving AZURE_BLOB_ENDPOINT / AZURE_STORAGE_CONNECTION_STRING / MSC_CONFIG
# from AZURE_SAS_TOKEN — the same credential path the trainer uses.
try:
    import vlm.data.energon_dataset  # noqa: F401  (import side effect: bootstrap)
    import multistorageclient as msc
except Exception as e:  # pragma: no cover - env/setup failure
    print(f"ERROR import {e}")
    sys.exit(2)


def main() -> int:
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "msc://azure/data/yiming/bee_stage2/train/train.jsonl"
    )
    try:
        client, path = msc.resolve_storage_client(url)
        info = client.info(path)
        size = int(getattr(info, "content_length", 0) or 0)
        # A 0-byte object means the blob exists but the upload is still flushing;
        # treat as not-ready so we never index an empty/partial jsonl.
        if size <= 0:
            print("MISSING")
            return 1
        print(f"FOUND {size}")
        return 0
    except FileNotFoundError:
        print("MISSING")
        return 1
    except Exception as e:
        # Includes Azure 404 wrappers (not always FileNotFoundError) and any
        # transient network/credential error — caller treats !=0,!=1 as "wait".
        msg = str(e).lower()
        if "not found" in msg or "404" in msg or "blobnotfound" in msg or "nosuchkey" in msg:
            print("MISSING")
            return 1
        print(f"ERROR {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
