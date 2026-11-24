"""Step 22: slide-sized charts for the 10-minute presentation.

Re-draws the report-pack figures (step 21) at 16:9 slide size with larger, sans-serif type, and
adds two slide-only charts (the data funnel and the user-history gain). Like step 21, nothing
here is presentation text: the authors write the slides.

Writes PNG (for PowerPoint/Keynote/Google Slides) and PDF to ``docs/presentation_pack/charts/``.

Examples::

    python scripts/22_presentation_pack.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, setup, stage  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.collections  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.text  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402

spec = importlib.util.spec_from_file_location("pack", Path(__file__).resolve().parent / "21_report_pack.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)

OUT = PROJECT_ROOT / "docs" / "presentation_pack"
WIDE, HALF = 11.5, 6.2            # inches: full-slide chart, half-slide chart (16:9 slide = 13.33 x 7.5)


def slide_style() -> None:
    pack.style()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "lines.linewidth": 2.4, "lines.markersize": 8, "savefig.dpi": 200,
    })


def funnel(s: ResultStore) -> None:
    ds = s.json("data_summary") or {}
    cl, sz, sp = ds.get("cleaning", {}), ds.get("sessionize", {}).get("5", {}), (ds.get("splits") or {}).get("5", {})
    steps = [("Raw sessions (7 months)", cl.get("sessions_in", 0)),
             ("10% session sample, cleaned", cl.get("sessions_out", 0)),
             ("Eligible at K=5 (>5 clicks, no buy yet)", sz.get("eligible", 0)),
             ("Training sessions", sp.get("train", {}).get("sessions", 0)),
             ("Test sessions (last weeks)", sp.get("test", {}).get("sessions", 0))]
    fig, ax = plt.subplots(figsize=(WIDE, 4.6))
    vals = [v for _, v in steps]
    ax.barh(range(len(steps)), vals, color=[pack.C["grey"]] * 2 + [pack.C["ML"]] * 3, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v / 1e6:,.2f} M" if v >= 1e6 else f"{v / 1e3:,.0f} k", va="center", fontsize=14)
    ax.set_xscale("log")
    ax.set_yticks(range(len(steps)), [n for n, _ in steps])
    ax.invert_yaxis()
    ax.set_xlim(1e5, max(vals) * 4)
    ax.set_xlabel("Sessions (log scale)")
    ax.grid(axis="y", visible=False)
    pack.save(fig, "s_data_funnel")


def user_history(s: ResultStore) -> None:
    seg = s.csv("user_history_segments")
    tests = s.csv("user_history_tests")
    if seg is None or tests is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    t = tests.sort("k")
    axes[0].bar([f"K={k}" for k in t["k"]], t["diff"], color=pack.C["good"], width=0.6)
    for i, v in enumerate(t["diff"]):
        axes[0].text(i, v + 0.002, f"+{v:.3f}", ha="center", fontsize=14)
    axes[0].set_ylabel("PR-AUC gain from user history")
    axes[0].set_ylim(0, float(t["diff"].max()) * 1.25)
    axes[0].grid(axis="x", visible=False)
    d = seg.filter(pl.col("k") == 5).sort("segment")
    x = range(d.height)
    w = 0.36
    axes[1].bar([i - w / 2 for i in x], d["pr_auc_base"], w, color=pack.C["grey"], label="session only")
    axes[1].bar([i + w / 2 for i in x], d["pr_auc_history"], w, color=pack.C["good"], label="+ user history")
    axes[1].set_xticks(list(x), [f"{g} users\n({n / d['sessions'].sum():.0%} of visits)"
                                 for g, n in zip(d["segment"], d["sessions"])])
    axes[1].set_ylabel("Test PR-AUC (K=5)")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(axis="x", visible=False)
    pack.save(fig, "s_user_history")


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    cfg = setup(p.parse_args(argv))
    s = ResultStore(cfg)
    # reuse the report figures at slide size; send their outputs to this pack
    pack.OUT, pack.COL, pack.FULL = OUT / "_unused", HALF, WIDE
    tables_off, real_save = pack.latex, pack.save
    pack.latex = lambda *a, **k: None

    def save(fig, name):
        if not name.startswith("s_"):            # a report figure: enlarge for the projector
            fig.set_size_inches(max(fig.get_size_inches()[0], HALF), max(fig.get_size_inches()[1] * 1.9, 3.6))
            for t in fig.findobj(matplotlib.text.Text):
                if t.get_text() and t.get_fontsize() < 12:
                    t.set_fontsize(12)
            for ln in fig.findobj(matplotlib.lines.Line2D):
                if ln.get_markersize() < 7:
                    ln.set_markersize(max(ln.get_markersize() * 2, 7))
                ln.set_linewidth(max(ln.get_linewidth() * 1.8, 1.5))
            for coll in fig.findobj(matplotlib.collections.LineCollection):   # error bars
                coll.set_linewidth(2.2)
            for pc in fig.findobj(matplotlib.collections.PathCollection):     # scatter dots
                pc.set_sizes([v * 3 for v in pc.get_sizes()])
        (OUT / "charts").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / "charts" / f"{name.replace('fig', 's_fig', 1)}.{ext}")
        plt.close(fig)

    pack.save = save
    slide_style()
    f = pack.Facts()
    with stage("Slide charts"):
        for fn in (pack.fig_rq1, pack.fig_rq2, pack.fig_rq3, pack.fig_rq4, pack.fig_extensions, pack.fig_segments,
                   pack.fig_business_agent):
            fn(s, f)
        funnel(s)
        user_history(s)
    pack.latex, pack.save = tables_off, real_save
    print(f"-> {(OUT / 'charts').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
