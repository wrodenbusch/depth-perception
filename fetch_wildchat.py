#!/usr/bin/env python
"""Fetch WildChat-1M (allenai/WildChat-1M, ODC-BY) into the HuggingFace cache. 14 parquet shards, ~3.1 GiB.

    HF_HOME=$HF_HOME .venv/bin/python fetch_wildchat.py

The source of the chat cell (``build_chat_pack.py``). It is REAL users' conversations with ChatGPT and
can contain personal data: nothing from it may be quoted into docs, commits, reports or figures --
aggregate statistics only.

Resumable, as ``fetch_web_shards.py``: ``hf_hub_download`` skips a file already complete in the cache.
Refuses to run without HF_HOME, or the download lands on the root disk.
"""
from __future__ import annotations
from topos_paths import HF_HOME  # noqa: E402

import os
import sys
import time

REPO = "allenai/WildChat-1M"


def main() -> None:
    if "HF_HOME" not in os.environ:
        sys.exit(f"[fetch] REFUSED: set HF_HOME={HF_HOME}, or this lands on /")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    names = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.endswith(".parquet"))
    print(f"[fetch] {len(names)} shards -> {os.environ['HF_HOME']}", flush=True)
    t0, total = time.time(), 0
    for i, name in enumerate(names, 1):
        for attempt in range(1, 6):
            try:
                p = hf_hub_download(REPO, name, repo_type="dataset")
                break
            except Exception as exc:            # a flaky transfer must not lose the batch
                if attempt == 5:
                    sys.exit(f"[fetch] GIVING UP on {name}: {exc}")
                time.sleep(min(60, 2 ** attempt))
        total += os.path.getsize(p)
        print(f"[fetch] {i}/{len(names)} {name}  {total/2**30:.2f} GiB  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
