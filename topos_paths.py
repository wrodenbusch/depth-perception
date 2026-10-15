"""Where data, runs and raw sources live. Set these environment variables; nothing is hard-coded.

    TOPOS_PACKS    built training packs (token streams + tree sidecars)   default ./work/packs
    TOPOS_RUNS     training runs: train.jsonl, checkpoints, analyses      default ./work/runs
    TOPOS_SOURCES  raw corpora the pack builders read                     default ./work/sources
    TOPOS_SITE     where make_appendix.py / refresh_controls.sh publish   default ./work/site
    HF_HOME        Hugging Face cache used by the fetch scripts           (set it yourself)

Every script and config refers to these names, so one machine's layout never leaks into the code.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORK = os.path.join(_HERE, "work")

PACKS = os.environ.get("TOPOS_PACKS", os.path.join(_WORK, "packs"))
RUNS = os.environ.get("TOPOS_RUNS", os.path.join(_WORK, "runs"))
SOURCES = os.environ.get("TOPOS_SOURCES", os.path.join(_WORK, "sources"))
SITE = os.environ.get("TOPOS_SITE", os.path.join(_WORK, "site"))
HF_HOME = os.environ.get("HF_HOME", os.path.join(_WORK, "hf_cache"))
REPO = _HERE
