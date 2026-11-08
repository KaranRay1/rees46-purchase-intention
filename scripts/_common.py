"""Shared command-line helpers for the numbered experiment scripts.

Every script accepts the same basic options::

    --mode synthetic|dev|full      override data.mode
    --set key.path=value           override any config value (repeatable)

and starts with ``cfg = setup(args)``. Running a script from the project root
(``python scripts/01_prepare_data.py``) works because this module puts the
project root on ``sys.path``.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, set_seed  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["synthetic", "dev", "full"], help="override data.mode")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set sessionize.prefix_k=[5] (repeatable)",
    )
    p.add_argument("--config", type=Path, default=None, help="alternative config.yaml")
    return p


def setup(args: argparse.Namespace) -> dict[str, Any]:
    """Load config with overrides and seed everything with the first project seed."""
    overrides = list(args.overrides)
    if args.mode:
        overrides.append(f"data.mode={args.mode}")
    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg["project"]["seeds"][0])
    return cfg


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Print a stage header and its duration."""
    print(f"\n== {name}", flush=True)
    t = time.perf_counter()
    yield
    print(f"   done in {time.perf_counter() - t:.1f}s", flush=True)


def print_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    """Small fixed-width table for terminal output (no extra dependencies)."""
    if not rows:
        print("   (empty)")
        return
    cols = list(columns or rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"   # 4 significant digits
        if isinstance(v, int):
            return f"{v:,}"
        return "" if v is None else str(v)

    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("   " + "  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print("   " + "  ".join(v.rjust(w) for v, w in zip(row, widths)))
