"""The assistant's knowledge base: project documents and result tables as text chunks.

Two kinds of source:

* **documents** (report, model and data cards, report updates, README, research notes),
  split at Markdown headings and then at paragraphs into chunks of at most
  ``MAX_CHARS`` characters, each prefixed with its document title and heading path;
* **fact sheets** written here from the result tables (``results/tables/*_<mode>.*``):
  one short, self-contained text per topic and K, so that a question such as
  "What PR-AUC does LightGBM reach after 5 clicks?" has one chunk that answers it.

Every chunk has a stable ``id`` (``source#n``) used for citations and for the
retrieval evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import PROJECT_ROOT, resolve_path

MAX_CHARS = 1200
DOCS = {
    "report": "docs/report.md",
    "report_updates": "docs/report_updates_{mode}.md",
    "model_card": "docs/model_card_{mode}.md",
    "data_card": "docs/data_card_{mode}.md",
    "research_blueprint": "docs/research_blueprint.md",
    "proposals_analysis": "docs/proposals_analysis.md",
    "readme": "README.md",
}
LABEL = {"rule_based": "Rule baseline", "logistic_regression": "Logistic regression", "random_forest": "Random forest",
         "lightgbm": "LightGBM", "mlp": "MLP", "tabnet": "TabNet", "gru": "GRU (sequence)"}
TEAM = {"rule_based": "machine learning (Merve Kara)", "logistic_regression": "machine learning (Merve Kara)",
        "random_forest": "machine learning (Merve Kara)", "lightgbm": "machine learning (Merve Kara)",
        "mlp": "deep learning (Karan Ray)", "tabnet": "deep learning (Karan Ray)", "gru": "deep learning (Karan Ray)"}


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str          # document key or table name
    title: str           # human-readable heading path
    text: str
    kind: str            # "document" | "facts"

    @property
    def body(self) -> str:
        """What retrieval indexes and the model reads."""
        return f"{self.title}\n{self.text}"


# --------------------------------------------------------------------------- documents
def _split_markdown(md: str) -> list[tuple[str, str]]:
    """(heading path, section text) at every #/##/### heading."""
    path: list[str] = []
    out, buf = [], []
    for line in md.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                out.append((" › ".join(path), "\n".join(buf).strip()))
            buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        out.append((" › ".join(path), "\n".join(buf).strip()))
    return out


def _pack(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split at blank lines (then at lines) into pieces of at most ``limit`` characters."""
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        pieces = [para] if len(para) <= limit else para.splitlines()
        for p in pieces:
            if cur and len(cur) + len(p) + 2 > limit:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        parts.append(cur)
    return parts


def document_chunks(mode: str, root: Path = PROJECT_ROOT) -> list[Chunk]:
    chunks = []
    for key, rel in DOCS.items():
        p = root / rel.format(mode=mode)
        if not p.exists():
            continue
        doc_title = key.replace("_", " ").title()
        n = 0
        for heading, section in _split_markdown(p.read_text()):
            for piece in _pack(section):
                chunks.append(Chunk(f"{key}#{n}", key, f"[{doc_title}] {heading}".strip(), piece, "document"))
                n += 1
    return chunks


# --------------------------------------------------------------------------- fact sheets
def _csv(tables: Path, stem: str, mode: str) -> pl.DataFrame | None:
    p = tables / f"{stem}_{mode}.csv"
    return pl.read_csv(p) if p.exists() else None


def _json(tables: Path, stem: str, mode: str) -> Any:
    p = tables / f"{stem}_{mode}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pair(pairs: pl.DataFrame | None, k: int, a: str, b: str) -> dict | None:
    if pairs is None:
        return None
    r = pairs.filter((pl.col("k") == k) & (((pl.col("model_a") == a) & (pl.col("model_b") == b))
                                          | ((pl.col("model_a") == b) & (pl.col("model_b") == a))))
    if not r.height:
        return None
    row = r.row(0, named=True)
    sign = 1 if row["model_a"] == a else -1
    lo, hi = sorted((sign * row["diff_ci_low"], sign * row["diff_ci_high"]))
    return {"diff": sign * row["pr_auc_diff"], "lo": lo, "hi": hi, "p": row["p_bootstrap_holm"]}


def fact_chunks(cfg: Mapping[str, Any]) -> list[Chunk]:
    from src.serving.recommend import recommended

    mode = cfg["data"]["mode"]
    tables = resolve_path(cfg, "tables_dir")
    data_name = {"full": "seven-month data (Oct 2019 – Apr 2020)", "dev": "October 2019 data",
                 "synthetic": "synthetic test data"}.get(mode, mode)
    out: list[Chunk] = []

    def add(source: str, title: str, lines: list[str]) -> None:
        out.append(Chunk(f"{source}#{sum(c.source == source for c in out)}", source,
                         f"[Results · {data_name}] {title}", "\n".join(lines), "facts"))

    rq1, pairs, eff = (_csv(tables, "rq1_model_comparison", mode), _csv(tables, "rq1_pairwise_tests", mode),
                       _csv(tables, "rq4_efficiency", mode))
    if rq1 is not None:
        for k in sorted(set(rq1["k"].to_list())):
            d = rq1.filter(pl.col("k") == k).sort("pr_auc_mean", descending=True)
            lines = [f"Test-period results after K={k} clicks ({k} events seen), mean of {d['n_seeds'].max()} "
                     f"training seeds. Buyer base rate {d['base_rate'][0]:.1%}."]
            for r in d.iter_rows(named=True):
                lines.append(f"- {LABEL.get(r['model'], r['model'])} [{TEAM.get(r['model'], '')}]: PR-AUC "
                             f"{r['pr_auc_mean']:.4f} (sd {r['pr_auc_std']:.4f}, 95% CI {r['pr_auc_ci_low']:.3f}–"
                             f"{r['pr_auc_ci_high']:.3f}), lift {r['pr_auc_lift_mean']:.2f}x over random, ROC-AUC "
                             f"{r['roc_auc_mean']:.3f}, F1 {r['f1_mean']:.3f}, precision {r['precision_mean']:.3f}, "
                             f"recall {r['recall_mean']:.3f}.")
            add("rq1_model_comparison", f"Model accuracy at K={k}", lines)
            learned = d.filter(pl.col("model") != "rule_based")
            ml = learned.filter(pl.col("model").is_in(["logistic_regression", "random_forest", "lightgbm"])).row(0, named=True)
            dl = learned.filter(pl.col("model").is_in(["mlp", "tabnet", "gru"])).row(0, named=True)
            t = _pair(pairs, k, ml["model"], dl["model"])
            a, b = learned.row(0, named=True), learned.row(1, named=True)
            t2 = _pair(pairs, k, a["model"], b["model"])
            lines = [f"Machine learning vs deep learning at K={k}: best ML model {LABEL[ml['model']]} "
                     f"(PR-AUC {ml['pr_auc_mean']:.4f}) vs best DL model {LABEL[dl['model']]} "
                     f"(PR-AUC {dl['pr_auc_mean']:.4f})."]
            if t:
                lines.append(f"Paired bootstrap difference {t['diff']:+.4f} (95% CI {t['lo']:+.4f} to {t['hi']:+.4f}), "
                             f"Holm-corrected p = {t['p']:.3g}: "
                             + ("statistically significant." if t["p"] < 0.05 else "not significant, a statistical tie."))
            if t2 and {a["model"], b["model"]} != {ml["model"], dl["model"]}:
                lines.append(f"The two most accurate models overall are {LABEL[a['model']]} and {LABEL[b['model']]}: "
                             f"difference {t2['diff']:+.4f}, p = {t2['p']:.3g}.")
            if eff is not None:
                ship, tied = recommended(rq1, pairs, eff, k)
                lines.append(f"Engine choice at K={k}: {LABEL.get(ship, ship)}, the fastest of the models statistically "
                             f"tied with the most accurate one (tied: {', '.join(LABEL.get(m, m) for m in tied)}).")
            add("rq1_ml_vs_dl", f"Machine learning vs deep learning at K={k}", lines)
    if eff is not None:
        for k in sorted(set(eff["k"].to_list())):
            d = eff.filter(pl.col("k") == k).sort("latency_b1_p95_ms")
            lines = [f"Speed and size at K={k} (single thread, one visit at a time; Apple M-series laptop)."]
            for r in d.iter_rows(named=True):
                lines.append(f"- {LABEL.get(r['model'], r['model'])}: p95 latency {r['latency_b1_p95_ms']:.3f} ms per visit, "
                             f"{r['throughput_b256_per_sec']:,.0f} visits/s in batches of 256, model size "
                             f"{r['model_size_mb']:.2f} MB, training time {r['train_time_sec_mean']:.0f} s.")
            add("rq4_efficiency", f"Latency and efficiency at K={k}", lines)
    steps = _csv(tables, "rq3_ablation", mode)
    if steps is not None:
        for k in sorted(set(steps["k"].to_list())):
            d = steps.filter(pl.col("k") == k).sort("model", "step")
            lines = [f"Feature-group ablation ladder at K={k}: G1 activity, G2 cart signals, G3 behavioural ratios, "
                     "G4 product/category, G5 price and time. PR-AUC when groups are added one at a time:"]
            for r in d.iter_rows(named=True):
                gain = f", gain {r['pr_auc_gain']:+.4f}" if r["pr_auc_gain"] is not None else ""
                lines.append(f"- {LABEL.get(r['model'], r['model'])} {r['feature_set']}: PR-AUC {r['pr_auc_mean']:.4f}{gain}.")
            add("rq3_ablation", f"Feature ablation at K={k}", lines)
    for k in (3, 5, 10):
        g = _csv(tables, f"rq5_importance_groups_k{k}", mode)
        if g is not None:
            lines = [f"Share of model attribution (SHAP; attention for TabNet) by feature group at K={k}:"]
            for r in g.sort("mean_share", descending=True).iter_rows(named=True):
                per = ", ".join(f"{LABEL.get(m, m)} {r[m]:.0%}" for m in g.columns if m not in ("feature", "mean_share"))
                lines.append(f"- {r['feature']}: mean {r['mean_share']:.0%} ({per}).")
            add("rq5_importance", f"Which feature groups matter at K={k}", lines)
    hyp = _csv(tables, "hypothesis_scoreboard", mode)
    if hyp is not None:
        add("hypotheses", "Hypothesis scoreboard",
            [f"- {r['id']} ({r['rq']}): {r['hypothesis']}. Verdict: {r['verdict']}. Evidence: {r['evidence']}."
             for r in hyp.iter_rows(named=True)])
    cal = _csv(tables, "calibration", mode)
    if cal is not None:
        for k in sorted(set(cal["k"].to_list())):
            d = cal.filter(pl.col("k") == k)
            lines = [f"Calibration at K={k} (Platt scaling fitted on the validation weeks, checked on the test period). "
                     "Raw scores are class-balanced rankings; calibrated scores are purchase probabilities."]
            for r in d.iter_rows(named=True):
                lines.append(f"- {LABEL.get(r['model'], r['model'])}: mean raw score {r['mean_score_raw']:.3f} vs buyer "
                             f"rate {r['buyer_rate']:.3f}; calibrated mean {r['mean_prob']:.3f}; ECE {r['ece_raw']:.3f} -> "
                             f"{r['ece']:.3f}; Brier {r['brier_raw']:.4f} -> {r['brier']:.4f}; the buyer threshold "
                             f"corresponds to a {r['threshold_prob']:.1%} purchase probability.")
            add("calibration", f"Score calibration at K={k}", lines)
    mon = _json(tables, "monitoring_summary", mode)
    if mon:
        lines = [f"Monitoring after training (validation and test weeks). PSI drift thresholds: moderate > "
                 f"{mon['psi_thresholds']['moderate']}, major > {mon['psi_thresholds']['major']}."]
        for k, ms in mon["models"].items():
            for m, v in ms.items():
                lines.append(f"- K={k} {LABEL.get(m, m)}: {v['weeks']} weeks, weekly PR-AUC {v['pr_auc_min']:.3f}–"
                             f"{v['pr_auc_max']:.3f}, lift {v['lift_min']:.2f}–{v['lift_max']:.2f}x.")
        for k, v in mon["drift"].items():
            lines.append(f"- Input drift K={k}: major {', '.join(v['major']) or 'none'}; moderate "
                         f"{', '.join(v['moderate']) or 'none'}.")
        add("monitoring", "Weekly accuracy and input drift", lines)
    monthly = _csv(tables, "monthly_profile", mode)
    if monthly is not None:
        add("data_monthly", "Shop traffic by month",
            ["Events, visits (sessions), buying visits and conversion per calendar month:"]
            + [f"- {r['month']}: {r['events']:,} events, {r['sessions']:,} visits, {r['buying_sessions']:,} buying "
               f"visits, conversion {r['conversion']:.2%}." for r in monthly.iter_rows(named=True)])
    return out


def build_corpus(cfg: Mapping[str, Any]) -> list[Chunk]:
    return document_chunks(cfg["data"]["mode"]) + fact_chunks(cfg)
