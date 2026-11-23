"""Show how far the long experiment runs (steps 11, 14, 15, 16, 17) have got.

Reads the log files in ``results/runs/`` and the list of running processes; it changes
nothing, so it is safe to run at any time, as often as you like.

    .venv/bin/python scripts/progress.py            # once
    .venv/bin/python scripts/progress.py --watch 60 # refresh every 60 s (Ctrl+C to stop)

Status: ``running`` (the Python script is working), ``queued`` (a waiting shell will start it
later), ``done`` (finished; the time is when its log was last written) or ``not started``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"

# step, script, log, what it does
STEPS = [
    ("17", "17_user_history.py", "user_history.log", "User history (G6) vs session-only features"),
    ("16", "16_transformer.py", "transformer.log", "Transformer vs GRU (3 K values x 3 seeds)"),
    ("15", "15_tune.py", "tuning.log", "Optuna tuning: LightGBM 40, MLP 15, GRU 10 trials"),
    ("14", "14_embeddings.py", "embeddings.log", "item2vec + text embeddings as features"),
    ("11", "11_rag_eval.py", "rag_eval.log", "RAG evaluation: Llama 3.1 8B vs Qwen3 14B"),
]


def processes() -> list[str]:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return out.splitlines()[1:]


def status(script: str, procs: list[str]) -> str:
    mine = [p for p in procs if script in p and "progress.py" not in p]
    if any(re.match(r"\S*python\S*\s", p, re.IGNORECASE) for p in mine):
        return "running"
    if mine:
        return "queued"
    return ""


def bar(frac: float, width: int = 24) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = int(round(frac * width))
    return "[" + "#" * full + "-" * (width - full) + f"] {frac * 100:5.1f}%"


def transformer(log: str, running: bool) -> tuple[float, str]:
    today = datetime.now().strftime("%Y%m%d")
    dirs = sorted(d.name for d in RUNS.glob(f"{today}-*_rq6_transformer_transformer_k*") if d.is_dir())
    finished = max(0, len(dirs) - (1 if running else 0))
    now = re.findall(r"== (K=\d+)", log)
    where = f"{now[-1]}, seed {len(dirs) - 3 * (len(now) - 1)}/3" if now and running else ""
    return finished / 9, f"{finished}/9 models trained" + (f" · now {where}" if where else "")


def tuning(log: str, running: bool) -> tuple[float, str]:
    plan = {"lightgbm": 40, "mlp": 15, "gru": 10}
    total = sum(plan.values()) + len(plan)            # + one 3-seed retrain per model
    trials = {m: 0 for m in plan}
    current = None
    for line in log.splitlines():
        m = re.search(r"== K=\d+ (\w+): .*Optuna", line)
        if m:
            current = m.group(1)
        t = re.search(r"trial (\d+)/(\d+)", line)
        if t and current in trials:
            trials[current] = int(t.group(1))
    retrained = len(re.findall(r"== .*retrain the best setting[^\n]*\n\s+done in", log))
    done = sum(trials.values()) + retrained
    detail = " · ".join(f"{m} {trials[m]}/{n}" for m, n in plan.items())
    best = re.findall(r"best val PR-AUC ([\d.]+) \(default ([\d.]+)\)", log)
    if best:
        detail += " · last best " + ", ".join(f"{b} vs default {d}" for b, d in best)
    return done / total, detail


def stages(log: str, expected: int, label: str) -> tuple[float, str]:
    done = len(re.findall(r"done in", log))
    heads = re.findall(r"== (.+)", log)
    now = heads[-1] if heads and len(heads) > done else ""
    return done / expected, f"{done}/{expected} {label}" + (f" · now: {now[:60]}" if now else "")


def show() -> None:
    procs = processes()
    print(f"\nExperiment progress at {datetime.now():%H:%M:%S}\n")
    for step, script, logname, what in STEPS:
        st = status(script, procs)
        log_path = RUNS / logname
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        if st == "queued" or (not st and not log):
            frac, detail = 0.0, "waiting for the step before it" if st else "not started"
        elif step == "16":
            frac, detail = transformer(log, st == "running")
        elif step == "15":
            frac, detail = tuning(log, st == "running")
        elif step == "17":
            frac, detail = stages(log, 5, "stages (sessions, G6, K=3, K=5, K=10)")
        elif step == "14":
            frac, detail = stages(log, 5, "stages (item2vec, text, K=3, K=5, K=10)")
        else:
            frac, detail = stages(log, 3, "stages (retrieval, Llama, Qwen)")
        if not st and log:
            ended = datetime.fromtimestamp(log_path.stat().st_mtime)
            failed = "Traceback" in log
            st = "FAILED" if failed else "done"
            frac = frac if failed else 1.0
            detail = f"{'stopped' if failed else 'finished'} at {ended:%H:%M} · see results/runs/{logname}"
        print(f"Step {step}  {what}")
        print(f"  {st or 'not started':<11} {bar(frac)}")
        print(f"  {detail}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watch", type=int, metavar="SECONDS", help="refresh every SECONDS (Ctrl+C to stop)")
    args = p.parse_args()
    if not args.watch:
        show()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            show()
            print(f"(refreshing every {args.watch} s · Ctrl+C to stop)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
