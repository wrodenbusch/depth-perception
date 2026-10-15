#!/usr/bin/env python
"""Append jobs to a shared runqueue jobs file, safely, while the queue is reading it.

    .venv/bin/python add_jobs.py $TOPOS_RUNS/controls/jobs.json new_jobs.json

``runqueue.py`` re-reads its jobs file every poll, so several producers can feed one queue (and so
one set of cards) instead of each starting its own queue and colliding on the same GPUs. Writers
take an exclusive lock and replace the file atomically, so the queue never reads a half-written
list. A job whose name is already present is refused rather than silently duplicated.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    target, new_path = sys.argv[1], sys.argv[2]
    new = json.load(open(new_path))
    with open(target + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        jobs = json.load(open(target)) if os.path.exists(target) else []
        have = {j["name"] for j in jobs}
        dup = [j["name"] for j in new if j["name"] in have]
        if dup:
            raise SystemExit(f"refusing: already queued: {dup}")
        for j in new:
            for k in ("name", "out", "log", "cmd"):
                if k not in j:
                    raise SystemExit(f"job missing {k!r}: {j}")
        jobs += new
        tmp = target + ".tmp"
        json.dump(jobs, open(tmp, "w"), indent=1)
        os.replace(tmp, target)
    print(f"added {len(new)} job(s) to {target}: {', '.join(j['name'] for j in new)}")


if __name__ == "__main__":
    main()
