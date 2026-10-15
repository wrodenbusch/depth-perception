#!/usr/bin/env python
"""Run a JSON job list, at most ``--max`` at once, one card each.

    nohup .venv/bin/python runqueue.py jobs.json --cards 4,5,6,7 --max 4 > queue.log 2>&1 &

A job is ``{"name": ..., "cmd": "<shell command>", "log": "<stdout path>"}``, optionally with
``"after": "<job name>"`` so it starts only once that job is ``done`` (used to extend a run in place).
The list is re-read every poll, so jobs can be appended while the queue runs.
State lives in ``<jobs>.state.json`` as ``{name: {status, card, pid, rc, ...}}``;
a job whose status is not ``pending`` (absent) is never started twice, so a
restarted queue resumes where it stopped rather than relaunching finished work.

WHY AT MOST FOUR. The box has eight A4000s and the author's rule for this project
is four cards at a time (2026-09-21). ``--max`` defaults to 4 and is refused
above it; raising the cap is a decision for the author, not a flag.
"""
import argparse
import json
import os
import subprocess
import time

HARD_MAX = 5   # raised from 4 on 2026-09-21 20:30 when the author said "go ahead and use a fifth card"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie of a dead queue is reaped by init; a live pid may still be one of ours exiting.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _out_dir(job: dict):
    if "out" in job:
        return job["out"]
    for tok in job["cmd"].split():
        if tok.startswith("--out_dir="):
            return tok.split("=", 1)[1]
    return None


def _last_step(job: dict):
    """The last step logged in the job's train.jsonl, or None."""
    out = _out_dir(job)
    if not out or not os.path.exists(os.path.join(out, "train.jsonl")):
        return None
    last = None
    for line in open(os.path.join(out, "train.jsonl")):
        try:
            last = json.loads(line).get("step", last)
        except json.JSONDecodeError:
            pass
    return last


def _expected_final(job: dict):
    for tok in job["cmd"].split():
        if tok.startswith("--max_iters="):
            return int(tok.split("=", 1)[1])
    return None


def _stalled(job: dict, stall_s: int) -> bool:
    """True when the job's train.jsonl exists and has not been written for stall_s seconds."""
    out = _out_dir(job)
    if not out:
        return False
    log = os.path.join(out, "train.jsonl")
    if not os.path.exists(log):
        return False                      # still compiling / first eval; the log appears at step 0
    return time.time() - os.path.getmtime(log) > stall_s


def _kill_tree(pid: int) -> None:
    """SIGTERM then SIGKILL the shell and its python child (Popen with shell=True)."""
    import signal
    kids = []
    try:
        kids = [int(x) for x in open(f"/proc/{pid}/task/{pid}/children").read().split()]
    except OSError:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for q in [pid] + kids:
            try:
                os.kill(q, sig)
            except ProcessLookupError:
                pass
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--cards", default="3,4,5,6")
    ap.add_argument("--max", type=int, default=HARD_MAX)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stall-s", type=int, default=900, dest="stall_s",
                    help="kill and requeue a job whose train.jsonl is idle this long (an eval+checkpoint takes <2 min)")
    ap.add_argument("--retries", type=int, default=2, help="retries for a job that exits non-zero")
    a = ap.parse_args()
    if a.max > HARD_MAX:
        raise SystemExit(f"--max {a.max} exceeds the project's cap of {HARD_MAX} cards")
    cards = [int(c) for c in a.cards.split(",")]
    state_path = a.jobs + ".state.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    running = {}  # name -> (Popen | int pid, card)
    # A queue restart re-attaches to jobs it started earlier that are still alive
    # (by pid), so their cards stay busy; a dead one with no recorded exit is orphaned.
    for name, s in state.items():
        if s.get("status") == "running":
            pid = s.get("pid")
            if pid and _alive(pid):
                running[name] = (pid, s["card"])
                print(f"[queue] re-attached {name} pid {pid} on card {s['card']}", flush=True)
            else:
                s["status"] = "orphaned"

    def save() -> None:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    save()
    while True:
        jobs = json.load(open(a.jobs))
        by_name = {j["name"]: j for j in jobs}
        for name, (p, card) in list(running.items()):
            if isinstance(p, int):            # re-attached: exit code is unknowable, presence is not
                rc = None if _alive(p) else 0
                if rc == 0:
                    # A re-attached job that died before its last step did not succeed
                    # (seen 2026-09-22: a CUDA "unspecified launch failure" ended a run at
                    # 33,520 of 54,440 and it was recorded as done). Infer from the log.
                    exp, last = _expected_final(by_name.get(name, {})), _last_step(by_name.get(name, {}))
                    if exp is not None and (last is None or last < exp):
                        rc = 1
            else:
                rc = p.poll()
            if rc is None and _stalled(by_name.get(name, {}), a.stall_s):
                # A training process whose log has not moved for --stall-s is hung
                # (seen 2026-09-21: autograd thread spinning, GPU idle, NVRM scrub
                # timeouts in the kernel log). Kill it and put it back in the queue;
                # the next start resumes from its ckpt.pt.
                pid = p if isinstance(p, int) else p.pid
                _kill_tree(pid)
                state[name].update(status="pending", stalled=state[name].get("stalled", 0) + 1, killed=time.time())
                del running[name]
                save()
                print(f"[queue] {name} STALLED (log idle > {a.stall_s}s); killed pid {pid}, requeued", flush=True)
                continue
            if rc is not None:
                retries = state[name].get("retries", 0)
                if rc != 0 and retries < a.retries:
                    # Transient failures (an OOM on one pathological batch, a driver hiccup)
                    # get a bounded number of retries; the retry resumes from ckpt.pt if one exists.
                    state[name].update(status="pending", rc=rc, retries=retries + 1, failed_at=time.time())
                    print(f"[queue] {name} exited rc={rc}; retry {retries + 1}/{a.retries} queued", flush=True)
                else:
                    state[name].update(status="done" if rc == 0 else "failed", rc=rc, ended=time.time())
                    print(f"[queue] {name} exited rc={rc}", flush=True)
                del running[name]
                save()
        busy = {c for _, c in running.values()}
        free = [c for c in cards if c not in busy]
        for j in jobs:
            if len(running) >= a.max or not free:
                break
            if j["name"] in state and state[j["name"]].get("status") != "pending":
                continue
            if j.get("after") and state.get(j["after"], {}).get("status") != "done":
                continue                      # a dependency: the named job must have finished cleanly
            card = free.pop(0)
            os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card))
            cmd = j["cmd"]
            out = _out_dir(j)
            if out and os.path.exists(os.path.join(out, "ckpt.pt")) and "--init_from=" not in cmd:
                cmd += " --init_from=resume"     # a retry continues from the last full checkpoint
            p = subprocess.Popen(
                cmd, shell=True, env=env, cwd=os.getcwd(),
                stdout=open(j["log"], "a"), stderr=subprocess.STDOUT,
            )
            running[j["name"]] = (p, card)
            prev = state.get(j["name"], {})
            state[j["name"]] = dict(status="running", card=card, pid=p.pid, started=time.time(), cmd=cmd,
                                    stalled=prev.get("stalled", 0), retries=prev.get("retries", 0))
            save()
            print(f"[queue] started {j['name']} on card {card} pid {p.pid}", flush=True)
        finished = all(state.get(j["name"], {}).get("status") in ("done", "failed", "orphaned") for j in jobs)
        if not running and finished:
            print("[queue] all jobs finished", flush=True)
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
