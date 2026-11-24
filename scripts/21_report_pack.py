"""Step 21: build the report evidence pack (figures, tables, facts, error-analysis worksheet).

Everything is generated from the result tables, so it can be rebuilt after any rerun. Nothing
here is report prose: the pack holds the *evidence* the authors write the paper from.

Writes to ``docs/report_pack/``:

* ``figures/``  paper figures (PDF for LaTeX/Word, PNG to preview), IEEE column width (3.5 in)
  or full width (7.16 in);
* ``tables/``   the key tables as LaTeX (booktabs) and CSV;
* ``facts.md``  every headline number with the table it comes from;
* ``error_analysis_sheet.csv``  test sessions where the systems fail or disagree, with their
  clicks, for structured manual error analysis (columns to fill in by hand).

Examples::

    python scripts/21_report_pack.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, setup, stage  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.config import PROJECT_ROOT, processed_dir_for_k, resolve_path  # noqa: E402
from src.serving.predictor import available_runs  # noqa: E402
from src.serving.results import ResultStore, test_predictions  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "report_pack"
COL, FULL = 3.5, 7.16                                  # IEEE column / page width (inches)
LABEL = {"rule_based": "Rule baseline", "logistic_regression": "Logistic reg.", "random_forest": "Random forest",
         "lightgbm": "LightGBM", "mlp": "MLP", "tabnet": "TabNet", "gru": "GRU", "transformer": "Transformer"}
ORDER = ["rule_based", "logistic_regression", "random_forest", "lightgbm", "mlp", "tabnet", "gru"]
FAMILY = {"rule_based": "baseline", "logistic_regression": "ML", "random_forest": "ML", "lightgbm": "ML",
          "mlp": "DL", "tabnet": "DL", "gru": "DL"}
# Okabe-Ito: colour-blind safe and distinguishable in greyscale print
C = {"ML": "#0072B2", "DL": "#D55E00", "baseline": "#7f7f7f", "good": "#009E73", "grey": "#9a9a9a", "ink": "#222222"}
KS = [3, 5, 10]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": "#555555", "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}")
    plt.close(fig)


def latex(df: pl.DataFrame, name: str, caption: str, label: str, fmt: dict | None = None) -> None:
    """Write ``df`` as a booktabs LaTeX table plus a CSV twin."""
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "tables" / f"{name}.csv")
    fmt = fmt or {}

    def cell(c, v):
        if v is None:
            return "--"
        if c in fmt:
            return fmt[c](v)
        if isinstance(v, float):
            return f"{v:.4f}"
        v = str(v)
        return v if "$" in v else v.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    cols = df.columns
    head = " & ".join(c.replace("_", r"\_").replace("%", r"\%") for c in cols)
    rows = [" & ".join(cell(c, v) for c, v in zip(cols, r)) + r" \\" for r in df.iter_rows()]
    tex = "\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                     r"\footnotesize", r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
                     head + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (OUT / "tables" / f"{name}.tex").write_text(tex)


class Facts:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def add(self, what: str, value: str, source: str) -> None:
        self.lines.append(f"- {what}: **{value}**  ·  `{source}`")

    def write(self, mode: str) -> None:
        head = ["# Facts sheet (auto-generated)", "",
                f"Every number below is read from `results/tables/*_{mode}.*` by `scripts/21_report_pack.py`. "
                "Cite figures from here or from the tables, never from memory. Rebuild after any rerun.", ""]
        (OUT / "facts.md").write_text("\n".join(head + self.lines) + "\n")


# --------------------------------------------------------------------------- figures + tables
def fig_rq1(s: ResultStore, f: Facts) -> pl.DataFrame:
    rq1 = s.csv("rq1_model_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharex=False)
    models = [m for m in ORDER if m != "rule_based"]
    for ax, k in zip(axes, KS):
        d = rq1.filter((pl.col("k") == k) & pl.col("model").is_in(models))
        d = d.with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(ORDER)}, default=99).alias("o")
                           ).sort("o")
        for i, r in enumerate(d.iter_rows(named=True)):
            col = C[FAMILY[r["model"]]]
            ax.errorbar(r["pr_auc_mean"], i, xerr=[[r["pr_auc_mean"] - r["pr_auc_ci_low"]],
                                                   [r["pr_auc_ci_high"] - r["pr_auc_mean"]]],
                        fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1, capsize=1.5)
        ax.set_yticks(range(d.height), [LABEL[m] for m in d["model"]] if k == 3 else [""] * d.height)
        ax.invert_yaxis()
        ax.set_xlim(float(d["pr_auc_ci_low"].min()) - 0.003, float(d["pr_auc_ci_high"].max()) + 0.003)
        rb = rq1.filter((pl.col("k") == k) & (pl.col("model") == "rule_based"))
        rule = f", rule baseline {rb['pr_auc_mean'][0]:.3f}" if rb.height else ""
        ax.set_title(f"K = {k} clicks\nbuyer rate {d['base_rate'][0]:.1%}{rule}", fontsize=7)
        ax.set_xlabel("Test PR-AUC")
    h = [plt.Line2D([], [], marker="o", ls="", color=C[x], label=lab) for x, lab in
         (("ML", "Machine learning (Merve Kara)"), ("DL", "Deep learning (Karan Ray)"))]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14), frameon=False)
    save(fig, "fig1_rq1_model_comparison")

    eff = s.csv("rq4_efficiency")
    f.section("RQ1: model comparison (test PR-AUC, mean of 3 seeds)")
    rows = []
    for k in KS:
        d = rq1.filter(pl.col("k") == k).sort("pr_auc_mean", descending=True)
        for r in d.iter_rows(named=True):
            f.add(f"K={k} {LABEL[r['model']]} PR-AUC (sd)", f"{r['pr_auc_mean']:.4f} ({r['pr_auc_std']:.4f})",
                  "rq1_model_comparison")
        f.add(f"K={k} buyer rate (test)", f"{d['base_rate'][0]:.4f}", "rq1_model_comparison")
    k5 = rq1.filter(pl.col("k") == 5).join(eff.filter(pl.col("k") == 5).select(
        "model", "latency_b1_p95_ms", "model_size_mb", "throughput_b1_per_sec"), on="model", how="left")
    k5 = k5.with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(ORDER)}, default=99).alias("o")
                         ).sort("o")
    t = pl.DataFrame({
        "Model": [LABEL[m] for m in k5["model"]],
        "PR-AUC": [f"{a:.4f} $\\pm$ {b:.4f}" for a, b in zip(k5["pr_auc_mean"], k5["pr_auc_std"])],
        "ROC-AUC": [f"{v:.4f}" for v in k5["roc_auc_mean"]],
        "F1": [f"{v:.4f}" for v in k5["f1_mean"]],
        "p95 ms": [f"{v:.3f}" for v in k5["latency_b1_p95_ms"]],
        "Size MB": [f"{v:.2f}" if v >= 0.01 else "<0.01" for v in k5["model_size_mb"]],
        "Train s": [f"{v:.0f}" if v >= 1 else "<1" for v in k5["train_time_sec_mean"]],
    })
    latex(t, "tab_main_k5", "Test results at $K=5$ (mean $\\pm$ sd over seeds 42, 7, 2026); latency per visit, "
          "batch size 1.", "tab:main")
    rows.append(t)
    return rq1


def fig_rq2(s: ResultStore, f: Facts) -> None:
    d = s.csv("rq2_by_k")
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for m in ORDER:
        r = d.filter(pl.col("model") == m).sort("k")
        if r.height:
            ax.plot(r["k"], r["pr_auc_lift_mean"], marker="o", ms=3, lw=1.2, color=C[FAMILY[m]],
                    ls="-" if m in ("gru", "random_forest", "lightgbm") else ":", label=LABEL[m])
    ax.set_xticks(KS)
    ax.set_xlabel("Clicks seen before predicting (K)")
    ax.set_ylabel("PR-AUC lift over buyer rate (×)")
    ax.legend(ncol=4, frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              columnspacing=0.8, handlelength=1.8)
    save(fig, "fig2_rq2_lift_by_k")
    f.section("RQ2: how early (PR-AUC lift = PR-AUC / buyer rate)")
    for r in d.filter(pl.col("model").is_in(["gru", "random_forest", "lightgbm"])).sort("model", "k").iter_rows(named=True):
        f.add(f"{LABEL[r['model']]} K={r['k']} PR-AUC / lift", f"{r['pr_auc_mean']:.4f} / {r['pr_auc_lift_mean']:.2f}",
              "rq2_by_k")


def fig_rq3(s: ResultStore, f: Facts) -> None:
    a = s.csv("rq3_ablation").filter(~pl.col("is_variant"))
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for m, col in (("lightgbm", C["ML"]), ("mlp", C["DL"])):
        for k, ls in ((3, ":"), (5, "-"), (10, "--")):
            r = a.filter((pl.col("model") == m) & (pl.col("k") == k)).sort("step")
            if r.height:
                ax.plot(r["step"], r["pr_auc_mean"], marker="o", ms=2.5, lw=1.1, ls=ls, color=col,
                        label=f"{LABEL[m]} K={k}")
    steps = a.filter((pl.col("model") == "lightgbm") & (pl.col("k") == 5)).sort("step")
    ax.set_xticks(steps["step"].to_list(), [x.split(":")[1] for x in steps["feature_set"]], rotation=20)
    ax.set_ylabel("Test PR-AUC")
    ax.set_xlabel("Feature groups (cumulative)")
    ax.legend(ncol=2, frameon=False, fontsize=5.5)
    save(fig, "fig3_rq3_feature_ladder")
    tests = s.csv("rq3_step_tests")
    f.section("RQ3: feature ablation (cumulative ladder)")
    for r in a.filter(pl.col("k") == 5).sort("model", "step").iter_rows(named=True):
        f.add(f"K=5 {LABEL[r['model']]} {r['feature_set']} PR-AUC", f"{r['pr_auc_mean']:.4f}", "rq3_ablation")
    for r in tests.filter((pl.col("k") == 5) & (pl.col("family") != "variant")).iter_rows(named=True):
        f.add(f"K=5 {LABEL[r['model']]} add {r['added']} gain [CI] Holm p",
              f"{r['pr_auc_gain']:+.4f} [{r['gain_ci_low']:+.4f}, {r['gain_ci_high']:+.4f}] p={r['p_bootstrap_holm']:.3f}",
              "rq3_step_tests")
    g = s.csv("rq5_importance_groups_k5")
    if g is not None:
        for r in g.iter_rows(named=True):
            f.add(f"K=5 mean SHAP share {r['feature']}", f"{r['mean_share']:.3f}", "rq5_importance_groups_k5")


def fig_rq4(s: ResultStore, f: Facts) -> None:
    e = s.csv("rq4_efficiency").filter(pl.col("k") == 5)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for r in e.filter(pl.col("model") != "rule_based").iter_rows(named=True):
        ax.scatter(r["latency_b1_p95_ms"], r["test_pr_auc_mean"], s=14 + 3 * np.sqrt(max(r["model_size_mb"], 0.001)),
                   color=C[FAMILY.get(r["model"], "baseline")], zorder=3)
        left = r["model"] == "mlp"
        ax.annotate(LABEL[r["model"]], (r["latency_b1_p95_ms"], r["test_pr_auc_mean"]), fontsize=6,
                    xytext=(-4 if left else 3, 2), textcoords="offset points", ha="right" if left else "left")
    ax.set_xscale("log")
    ax.set_xlabel("p95 latency per visit, batch 1 (ms, log scale); dot size ~ model size")
    ax.set_ylabel("Test PR-AUC (K=5)")
    lo = float(e.filter(pl.col("model") != "rule_based")["test_pr_auc_mean"].min())
    ax.set_ylim(lo - 0.006, float(e["test_pr_auc_mean"].max()) + 0.003)
    save(fig, "fig4_rq4_accuracy_vs_latency")
    f.section("RQ4: efficiency (K=5)")
    for r in e.sort("latency_b1_p95_ms").iter_rows(named=True):
        f.add(f"{LABEL[r['model']]} p95 latency / size / train time / throughput",
              f"{r['latency_b1_p95_ms']:.4f} ms / {r['model_size_mb']:.3f} MB / {r['train_time_sec_mean']:.1f} s / "
              f"{r['throughput_b1_per_sec']:,.0f} per s", "rq4_efficiency")


def fig_extensions(s: ResultStore, f: Facts) -> None:
    """Forest plot: what each extension adds (paired difference with 95% CI, K=5)."""
    rows = []
    uh = s.csv("user_history_tests")
    if uh is not None:
        r = uh.filter(pl.col("k") == 5).row(0, named=True)
        rows.append(("+ user history (G6), LightGBM", r["diff"], r["lower"], r["upper"], r["significant"]))
    em = s.csv("embeddings_tests")
    if em is not None:
        r = em.filter((pl.col("k") == 5) & (pl.col("feature_set") == "item2vec+text")).row(0, named=True)
        rows.append(("+ item2vec & text embeddings", r["diff"], r["lower"], r["upper"], r["significant"]))
    tu = s.csv("tuning")
    if tu is not None:
        for r in tu.iter_rows(named=True):
            rows.append((f"Optuna tuning, {LABEL[r['model']]}", r["diff"], r["lower"], r["upper"], bool(r["significant"])))
    st_ = s.csv("stacking_tests")
    if st_ is not None:
        r = st_.filter((pl.col("k") == 5) & (pl.col("ensemble") == "stack_all")).row(0, named=True)
        rows.append(("Stacking all models vs best single", r["diff"], r["lower"], r["upper"], r["significant"]))
    tr = s.csv("transformer_tests")
    if tr is not None:
        r = tr.filter((pl.col("k") == 5) & (pl.col("vs") == "gru")).row(0, named=True)
        rows.append(("Transformer vs GRU", r["diff"], r["lower"], r["upper"], r["significant"]))
    fig, ax = plt.subplots(figsize=(COL, 0.45 + 0.27 * len(rows)))
    for i, (_lab, d, lo, hi, sig) in enumerate(rows):
        col = C["good"] if sig and d > 0 else (C["DL"] if sig else C["grey"])
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1, capsize=1.5)
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.004, linscale=1.2)
    ticks = [0, 0.001, 0.002, 0.004, 0.01, 0.03, 0.06]
    ax.set_xticks(ticks, ["0", ".001", ".002", ".004", ".01", ".03", ".06"])
    ax.set_xlim(-0.0015, 0.08)
    ax.set_xlabel("Change in test PR-AUC at K=5 (95% CI; axis linear to .004, log above)")
    save(fig, "fig5_extensions_forest")
    f.section("Extensions at K=5 (paired bootstrap; green in fig5 = significant after Holm)")
    for lab, d, lo, hi, sig in rows:
        f.add(lab, f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] {'significant' if sig else 'n.s.'}", "see fig5 sources")
    t = pl.DataFrame({"Experiment": [r[0] for r in rows], "Gain": [r[1] for r in rows],
                      "CI low": [r[2] for r in rows], "CI high": [r[3] for r in rows],
                      "Sig. (Holm)": ["yes" if r[4] else "no" for r in rows]})
    latex(t, "tab_extensions_k5", "Effect of each extension on test PR-AUC at $K=5$ (paired bootstrap, 95\\% CI).",
          "tab:ext", {"Gain": lambda v: f"{v:+.4f}", "CI low": lambda v: f"{v:+.4f}", "CI high": lambda v: f"{v:+.4f}"})
    if uh is not None:
        for r in uh.iter_rows(named=True):
            f.add(f"User history gain K={r['k']}", f"{r['diff']:+.4f} [{r['lower']:+.4f}, {r['upper']:+.4f}]",
                  "user_history_tests")
        seg = s.csv("user_history_segments")
        if seg is not None:
            for r in seg.filter(pl.col("k") == 5).iter_rows(named=True):
                f.add(f"User history K=5, {r['segment']} users (share of test visits n={r['sessions']:,})",
                      f"{r['pr_auc_base']:.4f} -> {r['pr_auc_history']:.4f}", "user_history_segments")
        gap = PROJECT_ROOT / "results" / "tables" / "user_history_gap_check_full.csv"
        if gap.exists():
            for r in pl.read_csv(gap).iter_rows(named=True):
                f.add(f"User history gain K=5 counting only sessions ended >= {r['min_gap_hours']} h before",
                      f"{r['gain']:+.4f}", "user_history_gap_check")


def fig_segments(s: ResultStore, f: Facts) -> None:
    m = s.csv("segment_metrics")
    if m is None:
        return
    d = m.filter((pl.col("k") == 5) & (pl.col("model") == "gru") & pl.col("dimension").is_in(
        ["price", "category", "user_type", "time_of_day"]))
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    labels, vals, cols = [], [], []
    for dim, col in (("price", C["ML"]), ("user_type", C["DL"]), ("time_of_day", C["good"]), ("category", C["grey"])):
        for r in d.filter(pl.col("dimension") == dim).sort("segment").iter_rows(named=True):
            labels.append(f"{r['segment']}")
            vals.append(r["pr_auc_lift"])
            cols.append(col)
    ax.barh(range(len(vals)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(vals)), labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("GRU PR-AUC lift over the segment's buyer rate (K=5)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C["ML"], C["DL"], C["good"], C["grey"])]
    ax.legend(h, ["price", "user type", "time of day", "category"], frameon=False, fontsize=6, ncol=4,
              loc="lower center", bbox_to_anchor=(0.4, 1.0), handlelength=1, columnspacing=0.8)
    save(fig, "fig6_segments_lift")
    disp = s.csv("segment_disparity")
    f.section("Segments / fairness (GRU)")
    for r in disp.filter(pl.col("model") == "gru").sort("k", "dimension").iter_rows(named=True):
        f.add(f"K={r['k']} {r['dimension']}: worst/best segment, lift ratio, TPR gap, flag-rate gap",
              f"{r['worst_segment']} / {r['best_segment']}, {r['lift_ratio']:.2f}, {r['tpr_gap']:.3f}, "
              f"{r['flag_rate_gap']:.3f}", "segment_disparity")
    tt = s.csv("segment_tests")
    if tt is not None and "significant" in tt.columns:
        f.add("GRU vs random forest: segments with a significant difference (Holm), all K",
              f"{int(tt['significant'].sum())} of {tt.height}", "segment_tests")
    for r in m.filter((pl.col("k") == 5) & (pl.col("model") == "gru") & (pl.col("dimension") == "user_type")
                      ).iter_rows(named=True):
        f.add(f"K=5 GRU {r['segment']} users: flag rate / lift", f"{r['flag_rate']:.3f} / {r['pr_auc_lift']:.2f}",
              "segment_metrics")


def fig_business_agent(s: ResultStore, f: Facts) -> None:
    bv = s.csv("business_value")
    if bv is not None:
        f.section("Business value (illustrative; assumptions in business_value_summary)")
        info = s.json("business_value_summary") or {}
        o = info.get("offer", {})
        f.add("Assumptions", f"discount {o.get('discount')}, uplift {o.get('uplift')}, order value "
              f"${o.get('aov', 0):,.0f}", "business_value_summary")
        for r in bv.filter(pl.col("model") == "gru").sort("k").iter_rows(named=True):
            f.add(f"K={r['k']} GRU net/month vs offer-to-all gain", f"${r['net']:,.0f} (+${r['gain_vs_blanket']:,.0f})",
                  "business_value")
        for k in KS:
            d = bv.filter((pl.col("k") == k) & (pl.col("model") != "rule_based"))
            if d.height:
                f.add(f"K={k} spread between ML/DL models", f"${float(d['net'].max() - d['net'].min()):,.0f}",
                      "business_value")
        for k, v in (info.get("spearman_pr_auc_vs_profit") or {}).items():
            f.add(f"K={k} Spearman PR-AUC rank vs profit rank", f"{v:.2f}", "business_value_summary")
    ae = s.csv("agent_eval_summary")
    if ae is not None:
        fig, ax = plt.subplots(figsize=(COL, 1.9))
        groups = [(m, q) for m in sorted(ae["model"].unique()) for q in ("standard", "tool")]
        x = np.arange(len(groups))
        for j, (sys_, col) in enumerate((("rag", C["ML"]), ("agent", C["DL"]))):
            v = [float(ae.filter((pl.col("system") == sys_) & (pl.col("model") == m) & (pl.col("question_set") == q)
                                 )["accuracy"][0]) for m, q in groups]
            ax.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=col, label="RAG" if sys_ == "rag" else "Agent (tools)")
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 0.5) * 0.38, vi + 0.02, f"{vi:.0%}", ha="center", fontsize=5.5)
        ax.set_xticks(x, [f"{m.split(':')[0]}\n{q} Qs" for m, q in groups], fontsize=6)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Answers correct")
        ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save(fig, "fig7_agent_vs_rag")
        f.section("LLM agent vs RAG (local Ollama models)")
        for r in ae.sort("model", "question_set", "system").iter_rows(named=True):
            f.add(f"{r['model']} {r['system']} on {r['question_set']} questions (n={r['n']}): accuracy / grounding / s",
                  f"{r['accuracy']:.3f} / {r['numeric_grounding']:.3f} / {r['seconds_mean']:.1f}", "agent_eval_summary")
        rag = s.csv("rag_answer_summary")
        ret = s.csv("rag_retrieval")
        if ret is not None:
            for r in ret.filter(pl.col("origin") == "all").iter_rows(named=True):
                f.add(f"Retrieval {r['retriever']} recall@5 / MRR@10", f"{r['recall@5']:.3f} / {r['mrr@10']:.3f}",
                      "rag_retrieval")
        if rag is not None:
            for r in rag.iter_rows(named=True):
                f.add(f"RAG step 11 {r['provider']}: accuracy / abstention / grounding",
                      f"{r['accuracy']:.3f} / {r['abstention']:.3f} / {r['numeric_grounding']:.3f}", "rag_answer_summary")


def data_and_robustness(s: ResultStore, f: Facts) -> None:
    ds = s.json("data_summary") or {}
    cl, sz, sp = ds.get("cleaning", {}), ds.get("sessionize", {}), ds.get("splits", {})
    f.section("Data")
    f.add("Raw events / sessions (7 months)", f"{cl.get('rows_in', 0):,} / {cl.get('sessions_in', 0):,}", "data_summary")
    f.add("After 10% session sample + cleaning: events / sessions / buying sessions",
          f"{cl.get('rows_out', 0):,} / {cl.get('sessions_out', 0):,} / {cl.get('purchase_sessions_out', 0):,}",
          "data_summary")
    for k, v in (cl.get("removed") or {}).items():
        f.add(f"Removed by cleaning rule '{k}'", f"{v:,}", "data_summary")
    rows = []
    for k in map(str, KS):
        z = sz.get(k, {})
        f.add(f"K={k}: too short / bought within prefix / eligible / buyer rate",
              f"{z.get('too_short', 0):,} / {z.get('purchase_within_prefix', 0):,} / {z.get('eligible', 0):,} / "
              f"{z.get('positive_rate', 0):.4f}", "data_summary")
        for part in ("train", "val", "test"):
            p = (sp.get(k) or {}).get(part, {})
            if p:
                rows.append({"K": int(k), "Split": part, "Sessions": p["sessions"], "Buyer rate": p["positive_rate"],
                             "From": p["first_start"][:10], "To": p["last_start"][:10]})
    if rows:
        t = pl.DataFrame(rows)
        latex(t, "tab_data_splits", "Chronological 70/15/15 split of eligible sessions (10\\% session sample).",
              "tab:data", {"Sessions": lambda v: f"{v:,}", "Buyer rate": lambda v: f"{v:.3f}"})
    sc = PROJECT_ROOT / "results" / "full100" / "tables" / "scale_comparison_k5.csv"
    if sc.exists():
        d = pl.read_csv(sc)
        f.section("Robustness: 10% sample vs 100% of sessions (K=5)")
        for r in d.iter_rows(named=True):
            f.add(f"{LABEL[r['model']]} PR-AUC 10% -> 100%", f"{r['pr_auc_10pct']:.4f} -> {r['pr_auc_100pct']:.4f}",
                  "full100/scale_comparison_k5")
        pw = PROJECT_ROOT / "results" / "full100" / "tables" / "rq1_pairwise_tests_full.csv"
        if pw.exists():
            r = pl.read_csv(pw).filter((pl.col("model_a") == "random_forest") & (pl.col("model_b") == "gru")).row(0, named=True)
            f.add("100%: random forest minus GRU [CI], Holm p",
                  f"{r['pr_auc_diff']:+.4f} [{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}], {r['p_bootstrap_holm']:.3f}",
                  "full100/rq1_pairwise_tests")
        latex(d.select(pl.col("model").replace_strict(LABEL, default=None).alias("Model"), "pr_auc_10pct", "pr_auc_100pct",
                       "change", "train_sec_10pct", "train_sec_100pct"),
              "tab_full_scale", "Generalisation to all sessions: 10\\% sample vs 100\\% at $K=5$.", "tab:scale")
    rt = s.csv("retraining_tests")
    if rt is not None:
        f.section("Robustness: drift and retraining (K=5, monthly replay)")
        for r in rt.filter(pl.col("significant")).iter_rows(named=True):
            f.add(f"{r['month']} {r['policy']} vs {r['vs']}", f"{r['diff']:+.4f} [{r['lower']:+.4f}, {r['upper']:+.4f}]",
                  "retraining_tests")
        f.add("Significant month x policy differences", f"{int(rt['significant'].sum())} of {rt.height}", "retraining_tests")
    cal = s.csv("calibration")
    if cal is not None:
        f.section("Calibration (Platt, fitted on validation)")
        for r in cal.filter(pl.col("model") == "gru").iter_rows(named=True):
            f.add(f"GRU K={r['k']} ECE raw -> calibrated", f"{r['ece_raw']:.4f} -> {r['ece']:.4f}", "calibration")
    cf = s.csv("conformal")
    if cf is not None:
        f.section("Conformal prediction (engine model)")
        for r in cf.filter(pl.col("engine") & (pl.col("k") == 5)).iter_rows(named=True):
            f.add(f"K=5 target {r['target_coverage']:.0%}: coverage / buyers' coverage / unsure share",
                  f"{r['coverage']:.3f} / {r['coverage_buyers']:.3f} / {r['share_unsure']:.3f}", "conformal")
    hs = s.csv("hypothesis_scoreboard")
    if hs is not None:
        f.section("Hypotheses")
        for r in hs.iter_rows(named=True):
            f.add(f"{r['id']} ({r['rq']}) {r['hypothesis']}", f"{r['verdict']}: {r['evidence']}", "hypothesis_scoreboard")
        latex(hs.select("id", "hypothesis", "verdict"), "tab_hypotheses", "Hypotheses and verdicts (Holm-corrected tests).",
              "tab:hyp")


# --------------------------------------------------------------------------- manual error analysis sheet
def error_sheet(cfg, k: int = 5, per_group: int = 12, seed: int = 42) -> int:
    runs = available_runs(cfg)
    thr = {}
    for m in ("gru", "random_forest"):
        r = runs.filter((pl.col("k") == k) & (pl.col("model") == m))
        if not r.height:
            return 0
        thr[m] = json.loads((resolve_path(cfg, "runs_dir") / r["run_id"][0] / "metrics.json").read_text())["threshold"]
    g, rf = test_predictions(cfg, k, "gru"), test_predictions(cfg, k, "random_forest")
    if g is None or rf is None or "user_session" not in g.columns:
        return 0
    d = g.select("user_session", "y_true", gru=pl.col("score")).join(
        rf.select("user_session", rf=pl.col("score")), on="user_session")
    d = d.with_columns(gru_flag=pl.col("gru") >= thr["gru"], rf_flag=pl.col("rf") >= thr["random_forest"])
    groups = {
        "both miss a buyer (FN)": (pl.col("y_true") == 1) & ~pl.col("gru_flag") & ~pl.col("rf_flag"),
        "both flag a non-buyer (FP)": (pl.col("y_true") == 0) & pl.col("gru_flag") & pl.col("rf_flag"),
        "only GRU right": (pl.col("gru_flag") == (pl.col("y_true") == 1)) & (pl.col("rf_flag") != (pl.col("y_true") == 1)),
        "only RF right": (pl.col("rf_flag") == (pl.col("y_true") == 1)) & (pl.col("gru_flag") != (pl.col("y_true") == 1)),
    }
    picks = []
    for name, cond in groups.items():
        sub = d.filter(cond)
        if sub.height:
            picks.append(sub.sample(min(per_group, sub.height), seed=seed).with_columns(group=pl.lit(name),
                                                                                         group_size=pl.lit(sub.height)))
    if not picks:
        return 0
    pk = pl.concat(picks)
    pdir = processed_dir_for_k(cfg, k)
    ev = (pl.scan_parquet(pdir / "prefix_events.parquet").filter(pl.col("user_session").is_in(pk["user_session"].implode()))
          .select("user_session", "pos", "event_type", "category_code", "brand", "price", "product_id").collect())
    clicks = ev.sort("user_session", "pos").group_by("user_session", maintain_order=True).agg(
        clicks=pl.concat_str([pl.col("event_type").cast(pl.String), pl.col("category_code").fill_null("unknown"),
                              pl.col("brand").fill_null("-"), pl.format("${}", pl.col("price").round(0)),
                              pl.format("p{}", pl.col("product_id"))], separator=" ").str.join(" | "))
    sess = pl.read_parquet(pdir / "sessions.parquet", columns=["user_session", "n_events_total", "first_purchase_pos"])
    out = (pk.join(clicks, on="user_session").join(sess, on="user_session", how="left")
           .select("group", "group_size", "user_session", "y_true", pl.col("gru").round(3), pl.col("rf").round(3),
                   "n_events_total", "first_purchase_pos", "clicks")
           .with_columns(error_type=pl.lit(""), likely_cause=pl.lit(""), notes=pl.lit(""))
           .sort("group", "user_session"))
    path = OUT / "error_analysis_sheet.csv"
    if path.exists():
        old = pl.read_csv(path, infer_schema_length=0)
        if any(old[c].drop_nulls().str.strip_chars().str.len_chars().sum() for c in ("error_type", "likely_cause", "notes")
               if c in old.columns):
            path = OUT / "error_analysis_sheet_new.csv"      # never overwrite hand-labelled work
            print(f"   {OUT.name}/error_analysis_sheet.csv has labels; wrote {path.name} instead")
    out.write_csv(path)
    return out.height


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    args = p.parse_args(argv)
    cfg = setup(args)
    s = ResultStore(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    f = Facts()
    with stage("Figures, tables and facts"):
        data_and_robustness(s, f)
        fig_rq1(s, f)
        fig_rq2(s, f)
        fig_rq3(s, f)
        fig_rq4(s, f)
        fig_extensions(s, f)
        fig_segments(s, f)
        fig_business_agent(s, f)
        f.write(cfg["data"]["mode"])
    with stage("Manual error-analysis worksheet (K=5, GRU vs random forest)"):
        n = error_sheet(cfg)
        print(f"   {n} sessions")
    print(f"-> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
