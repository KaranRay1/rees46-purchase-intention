"""Step 23: technical inventory of the project as a PDF (data, methods, algorithms, code, tools).

A reference sheet for the team and the examiner: what the project uses and where it lives.
Hyperparameters come from ``config/config.yaml``, numbers from the result tables, and the
module/function reference is read from the code's own docstrings, so the document follows
the code after any change. It is a technical reference, not report text.

Writes ``docs/project_inventory/project_inventory.html`` and, if Google Chrome or Chromium is
installed, ``project_inventory.pdf`` (headless print).

Examples::

    python scripts/23_project_inventory.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import ast
import html
import importlib.metadata as md
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.features.user_history import HISTORY_FEATURES  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "project_inventory"
DATA_URL = "https://data.rees46.com/datasets/marketplace/"
KAGGLE_URL = "https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store"
NAMES = {"rule_based": "Rule baseline (cart > 0)", "logistic_regression": "Logistic regression",
         "random_forest": "Random forest", "lightgbm": "LightGBM", "mlp": "MLP", "tabnet": "TabNet",
         "gru": "GRU", "transformer": "Transformer encoder"}
LIBS = [("polars", "data processing (lazy, streaming)"), ("pyarrow", "Parquet storage"),
        ("pandas", "interop with sklearn/SHAP"), ("numpy", "numerics"),
        ("scikit-learn", "LR, RF, metrics, PCA/SVD"), ("lightgbm", "gradient boosting"),
        ("torch", "MLP, GRU, Transformer"), ("pytorch-tabnet", "TabNet"),
        ("scipy", "statistics, sparse matrices"), ("statsmodels", "McNemar test"),
        ("shap", "explanations"), ("optuna", "Bayesian tuning (TPE)"), ("psutil", "memory measurement"),
        ("matplotlib", "report and slide figures"), ("seaborn", "notebook figures"),
        ("streamlit", "dashboard"), ("plotly", "dashboard charts"), ("fastapi", "REST API"),
        ("uvicorn", "API server"), ("httpx", "Ollama client, downloader"), ("pyyaml", "configuration"),
        ("pytest", "tests"), ("jupyterlab", "notebooks")]


def esc(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list], cls: str = "") -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def params(d: dict, skip=("tuning_grid", "monitor")) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items() if k not in skip)


def first_line(doc: str | None) -> str:
    text = (doc or "").strip().split("\n\n")[0].replace("\n", " ")
    return text.replace("``", "").replace(":class:", "").replace(":func:", "")


def module_reference() -> str:
    parts = []
    files = sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted((PROJECT_ROOT / "app").rglob("*.py"))
    for p in files:
        tree = ast.parse(p.read_text())
        items = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
                kind = "class" if isinstance(n, ast.ClassDef) else "def"
                items.append([f"<code>{kind} {esc(n.name)}</code>", esc(first_line(ast.get_docstring(n))[:220])])
        if not items:
            continue
        rel = p.relative_to(PROJECT_ROOT)
        parts.append(f'<h3><code>{esc(rel)}</code></h3><p class="mod">{esc(first_line(ast.get_docstring(tree)))}</p>'
                     + table(["Name", "Purpose (from its docstring)"], items, "ref"))
    return "".join(parts)


def scripts_table() -> str:
    rows = []
    for p in sorted((PROJECT_ROOT / "scripts").glob("[0-9][0-9]_*.py")):
        doc = first_line(ast.get_docstring(ast.parse(p.read_text())))
        rows.append([f"<code>{esc(p.name)}</code>", esc(doc.split(":", 1)[-1].split("—", 1)[-1].strip())])
    return table(["Script", "What it does"], rows)


def build(cfg) -> str:
    s = ResultStore(cfg)
    ds = s.json("data_summary") or {}
    cl, sz, sp = ds.get("cleaning", {}), ds.get("sessionize", {}), ds.get("splits", {})
    m, f, pp = cfg["models"], cfg["features"], cfg["preprocessing"]
    sec = []

    # 1 dataset
    removed = cl.get("removed", {}) if isinstance(cl.get("removed"), dict) else {}
    rule_rows = [[esc(k.replace("_", " ")), f"{v:,}"] for k, v in removed.items()]
    k_rows = []
    for k in cfg["sessionize"]["prefix_k"]:
        z, t = sz.get(str(k), {}), sp.get(str(k), {})
        k_rows.append([f"K={k}", f"{z.get('eligible', 0):,}", f"{z.get('positive_rate', z.get('buyer_rate', 0)):.2%}",
                       f"{t.get('train', {}).get('sessions', 0):,}", f"{t.get('val', {}).get('sessions', 0):,}",
                       f"{t.get('test', {}).get('sessions', 0):,}"])
    sp5 = sp.get("5", {})
    sec.append(f"""
<h2>1. Dataset</h2>
<p><b>REES46 "eCommerce behavior data from multi category store"</b>: anonymised clickstream of a
large online marketplace, one row per event. Public, no account needed:
<a href="{DATA_URL}">{DATA_URL}</a> (files <code>2019-Oct.csv.gz</code> … <code>2020-Apr.csv.gz</code>; also on
<a href="{KAGGLE_URL}">Kaggle</a>). Downloaded by <code>scripts/00_download_data.py</code>.</p>
{table(["Property", "Value"], [
    ["Period", "1 Oct 2019 – 30 Apr 2020 (7 monthly CSV files, about 53 GB uncompressed)"],
    ["Raw events / sessions", f"{cl.get('rows_in', 0):,} / {cl.get('sessions_in', 0):,}"],
    ["Columns", ", ".join(f"<code>{c}</code> ({t})" for c, t in cfg['data']['schema'].items())],
    ["Event types", ", ".join(cfg['data']['event_types']) + " (this store logs no remove_from_cart)"],
    ["Sample used for experiments", f"{cfg['data'].get('session_sample_fraction') or 1:.0%} of whole sessions "
     f"(seed {cfg['project']['seeds'][0]}); a 100% run validates the main result"],
    ["After sample + cleaning", f"{cl.get('rows_out', 0):,} events, {cl.get('sessions_out', 0):,} sessions"],
])}
<h3>Cleaning rules (<code>src/data/loader.py</code>)</h3>
<p>Duplicate rows dropped, null sessions dropped, price ≤ {cfg['data']['cleaning']['min_price']} dropped,
sessions longer than {cfg['data']['cleaning']['max_session_duration_hours']} h dropped as logging artefacts,
missing category/brand set to "unknown". Category labels harmonised across the December 2019 category-tree
change (<code>src/data/categories.py</code>: each new category id takes the old label most of its products
agree on, threshold {cfg['data']['cleaning']['harmonise_categories']['min_share']}).</p>
{table(["Rows removed by rule", "Count"], rule_rows) if rule_rows else ""}
<h3>Sessions, prefixes and target (<code>src/data/sessionize.py</code>)</h3>
<p>Each session is cut after its first <b>K ∈ {cfg['sessionize']['prefix_k']}</b> events. Only sessions with
more than K events are kept, sessions that buy within the first K events are removed, and the target is
<code>{cfg['sessionize']['target_column']}</code> = 1 if a purchase happens <i>after</i> event K. Features use
the first K events only, so nothing from the future leaks in.</p>
<h3>Split (<code>src/data/split.py</code>)</h3>
<p>Chronological by session start, {cfg['split']['train']:.0%} / {cfg['split']['val']:.0%} /
{cfg['split']['test']:.0%} train / validation / test, no shuffling; sessions crossing a boundary are purged.
At K=5: train {sp5.get('train', {}).get('first_start', '')[:10]} → {sp5.get('train', {}).get('last_start', '')[:10]},
validation to {sp5.get('val', {}).get('last_start', '')[:10]}, test to {sp5.get('test', {}).get('last_start', '')[:10]}.
The test weeks are never used for training, tuning, thresholds or calibration.</p>
{table(["K", "Eligible sessions", "Buyer rate", "Train", "Validation", "Test"], k_rows)}
""")

    # 2 features
    grp_rows = [[esc(g), ", ".join(f"<code>{c}</code>" for c in cols)] for g, cols in f["groups"].items()]
    grp_rows.append(["G6_user_history (extension, step 17)",
                     ", ".join(f"<code>{c}</code>" for c in HISTORY_FEATURES)
                     + ": from the user's earlier sessions, using only sessions that ended before this visit "
                     "started (<code>src/features/user_history.py</code>)"])
    grp_rows.append(["Embeddings (extension, step 14)",
                     "mean item2vec vector of visited products, mean text vector, coverage, coherence "
                     "(<code>src/features/embeddings.py</code>)"])
    prof = [[esc(n), esc(params(v))] for n, v in pp["profiles"].items()]
    sec.append(f"""
<h2>2. Features</h2>
<p>Aggregated features of the K-event prefix for the tabular models (<code>src/features/build_features.py</code>),
grouped for the ablation ladder (RQ3). Categorical: {", ".join(f["categorical"])}; top {f["top_n_categories"]}
categories kept, others → "other".</p>
{table(["Group", "Features"], grp_rows)}
<p><b>Sequences for the GRU/Transformer</b> (<code>src/features/sequences.py</code>): per event, embedded
event type (dim {cfg['sequences']['categorical']['event_type']['embedding_dim']}) and main category
(dim {cfg['sequences']['categorical']['main_category']['embedding_dim']}), plus standardised log price and
log gap since the previous event; exactly K steps, no padding.</p>
<p><b>Preprocessing</b> (<code>src/features/preprocessing.py</code>), fitted on train only; log1p on
heavy-tailed features ({", ".join(pp["log1p_features"])}).</p>
{table(["Profile", "Steps"], prof)}
<p>Profile per model: {", ".join(f"{NAMES.get(k, k)} → {v}" for k, v in pp["model_profiles"].items())}.</p>
""")

    # 3 models
    model_rows = [
        ["Baseline", NAMES["rule_based"], "hand rule: predict buyer if the prefix has a cart event", "–", "—", "Both"],
        ["ML", NAMES["logistic_regression"], "linear model, L2-regularised, sigmoid output",
         "scikit-learn", esc(params(m["logistic_regression"])), "Merve"],
        ["ML", NAMES["random_forest"], "bagged decision trees, random feature subsets (Breiman 2001)",
         "scikit-learn", esc(params(m["random_forest"])), "Merve"],
        ["ML", NAMES["lightgbm"], "gradient-boosted trees, leaf-wise growth, histogram splits (Ke et al. 2017)",
         "lightgbm", esc(params(m["lightgbm"])), "Merve"],
        ["DL", NAMES["mlp"], "feed-forward network, batch norm, dropout, Adam", "PyTorch",
         esc(params(m["mlp"])), "Karan"],
        ["DL", NAMES["tabnet"], "attentive feature selection over sequential decision steps (Arik & Pfister 2021)",
         "pytorch-tabnet", esc(params(m["tabnet"])), "Karan"],
        ["DL", NAMES["gru"], "gated recurrent unit over the K click events (Cho et al. 2014; Hidasi et al. 2016)",
         "PyTorch", esc(params(m["gru"])), "Karan"],
        ["DL ext.", NAMES["transformer"], "self-attention encoder over the clicks (Vaswani et al. 2017), step 16",
         "PyTorch", esc(params(m["transformer"])), "Karan"],
    ]
    sec.append(f"""
<h2>3. Models and algorithms</h2>
{table(["Family", "Model", "Algorithm", "Library", "Settings (config.yaml)", "Owner"], model_rows, "models")}
<p><b>Training policy for every model:</b> class weights "{cfg['imbalance']['class_weight']}" for the ≈10% buyers;
early stopping on validation PR-AUC (LightGBM, MLP, TabNet, GRU, Transformer); decision threshold chosen to
maximise F1 on validation; each model × K run with seeds {cfg['project']['seeds']}; CPU only so latencies are
comparable. Shared code: <code>src/models/base.py</code> (common interface and factory),
<code>src/models/torch_base.py</code> (PyTorch training loop).</p>
""")

    # 4 evaluation
    ev = cfg["evaluation"]
    sec.append(f"""
<h2>4. Evaluation and statistics</h2>
{table(["Method", "Details", "Code"], [
    ["Metrics", "PR-AUC (primary; average precision), ROC-AUC, precision, recall, F1, macro-F1; "
     "PR-AUC lift = PR-AUC / buyer rate", "<code>src/evaluation/metrics.py</code>"],
    ["Confidence intervals", f"bootstrap, {ev['bootstrap']['n_resamples']} resamples, "
     f"{ev['bootstrap']['confidence']:.0%}", "<code>src/evaluation/statistics.py</code>"],
    ["Model vs model", "paired bootstrap of the PR-AUC difference on the same test sessions; McNemar test on "
     "thresholded predictions", "<code>statistics.py</code>"],
    ["Multiple comparisons", f"Holm–Bonferroni correction, α = {ev['alpha']}", "<code>statistics.py</code>"],
    ["Seeds", "mean and standard deviation over 3 seeds", "<code>statistics.py</code>"],
    ["Efficiency (RQ4)", f"latency p50/p95/p99 at batch sizes {cfg['efficiency']['batch_sizes']} "
     f"({cfg['efficiency']['n_repeats']} repeats, {cfg['efficiency']['n_threads']} thread), throughput, model size, "
     "training time, peak memory", "<code>src/evaluation/efficiency.py</code>"],
    ["Explainability (RQ5)", "SHAP: TreeExplainer (RF, LightGBM), LinearExplainer (LR), DeepExplainer / "
     "GradientExplainer (MLP, TabNet); group shares, model agreement, SHAP by error group",
     "<code>src/explainability/shap_analysis.py</code>"],
    ["Error analysis", "true/false positives/negatives profiled by session length and behaviour; manual "
     "labelling worksheet", "<code>scripts/06_explain.py</code>, <code>21_report_pack.py</code>"],
])}
""")

    # 5 extensions
    sec.append(f"""
<h2>5. Advanced methods and extensions</h2>
{table(["Topic", "Method", "Code"], [
    ["Calibration", "Platt scaling fitted on validation; Brier score, expected calibration error (ECE), "
     "reliability curves", "<code>src/serving/calibration.py</code>, step 10"],
    ["Conformal prediction", "split conformal, class-conditional (Mondrian) label sets with a coverage "
     "guarantee; weekly coverage", "<code>src/evaluation/conformal.py</code>, step 12"],
    ["Stacking", "meta logistic regression on logit scores of all models; mean-logit and mean-rank "
     "averaging; error overlap between models (Wolpert 1992)", "<code>src/evaluation/stacking.py</code>, step 12"],
    ["Drift and retraining", "population stability index (PSI) for inputs; monthly replay of frozen, "
     "expanding, 3-month and 1-month window retraining (LightGBM)",
     "<code>src/evaluation/monitoring.py</code>, steps 07, 13"],
    ["Embeddings", "item2vec as truncated SVD of the PPMI co-occurrence matrix (Levy & Goldberg 2014); "
     "product text embedded with nomic-embed-text + PCA", "<code>src/features/embeddings.py</code>, step 14"],
    ["Hyperparameter tuning", "Optuna TPE sampler (Bergstra et al. 2011) on the validation weeks",
     "<code>scripts/15_tune.py</code>"],
    ["Transformer", "self-attention encoder vs the GRU on the same click sequences",
     "<code>src/models/sequence_models.py</code>, step 16"],
    ["User history (G6)", "as-of joins (<code>join_asof</code>) over the full event log, 30-day window; "
     "1 h / 24 h gap check against leakage", "<code>src/features/user_history.py</code>, step 17"],
    ["Business value", "profit of a discount-suppression campaign; cutoff chosen on one half of the test "
     "visits and measured on the other; bootstrap error bars; sensitivity grid",
     "<code>src/evaluation/business_value.py</code>, step 18"],
    ["Segments / fairness", "PR-AUC lift, flag rate, TPR/FPR and calibration gap per segment ("
     "product category, price level, cart in the first clicks, time of day, weekday or weekend, new or returning user); disparity ratios",
     "<code>src/evaluation/segments.py</code>, step 19"],
    ["RAG analyst", "BM25 (sparse) + nomic-embed-text (dense) retrieval fused by reciprocal rank fusion; "
     "answers by a local LLM (Llama 3.1 8B or Qwen3 14B via Ollama) with citations and a number check",
     "<code>src/llm/retrieval.py</code>, <code>rag.py</code>, step 11"],
    ["Tool-using agent", "function-calling loop (max 5 steps) with tools that read result tables, compare "
     "models, score sessions and value campaigns; compared with plain RAG",
     "<code>src/llm/agent.py</code>, step 20"],
])}
""")

    # 6 results
    rq1 = s.csv("rq1_model_comparison")
    eff = s.csv("rq4_efficiency")
    if rq1 is not None:
        d = rq1.filter(pl.col("k") == 5).sort("pr_auc_mean", descending=True)
        e = {} if eff is None else {r["model"]: r for r in eff.filter(pl.col("k") == 5).iter_rows(named=True)}
        rows = [[NAMES.get(r["model"], r["model"]), f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
                 f"{r['roc_auc_mean']:.4f}", f"{r['f1_mean']:.4f}", f"{r['pr_auc_lift_mean']:.2f}×",
                 f"{e[r['model']]['latency_b1_p95_ms']:.4f}" if r["model"] in e else "–",
                 f"{e[r['model']]['model_size_mb']:.3f}" if r["model"] in e else "–"]
                for r in d.iter_rows(named=True)]
        sec.append("<h2>6. Headline results (test set, K=5, mean of 3 seeds)</h2>"
                   + table(["Model", "PR-AUC ± sd", "ROC-AUC", "F1", "Lift", "p95 latency (ms)", "Size (MB)"], rows)
                   + f"<p>Test buyer rate {d['base_rate'][0]:.2%}. All numbers with sources: "
                     "<code>docs/report_pack/facts.md</code>; every table is in <code>results/tables/</code>.</p>")

    # 7 software
    lib_rows = []
    for name, use in LIBS:
        try:
            v = md.version(name)
        except md.PackageNotFoundError:
            v = "not installed"
        lib_rows.append([f"<code>{name}</code>", v, use])
    sec.append(f"""
<h2>7. Software, tools and deliverables</h2>
<p>Python {sys.version.split()[0]}. Libraries (installed versions):</p>
{table(["Library", "Version", "Used for"], lib_rows, "libs")}
{table(["Component", "Details"], [
    ["Configuration", "<code>config/config.yaml</code>: single source of truth; a snapshot is saved with every run"],
    ["Run logging", "<code>src/utils/results_logger.py</code>: one row per run in <code>results/tables/all_runs.csv</code> "
     "(lightweight replacement for MLflow)"],
    ["Dashboard", "Streamlit app <code>app/main.py</code>: Overview, Sales & traffic, Product catalogue, Machine "
     "learning, Deep learning, Model comparison, Advanced ML, Live scoring, Campaign planner, Monitoring, Analyst "
     "assistant"],
    ["REST API", "FastAPI <code>src/api/app.py</code>: <code>GET /health</code>, <code>GET /models</code>, "
     "<code>POST /score</code>, <code>POST /score/all</code>"],
    ["Containers", "<code>Dockerfile</code>, <code>docker-compose.yml</code>"],
    ["Local LLMs", "Ollama: <code>llama3.1:8b</code>, <code>qwen3:14b</code>, <code>nomic-embed-text</code> "
     "(components of the system under study)"],
    ["Notebooks", "<code>notebooks/</code>: exploratory and analysis notebooks"],
    ["Tests", f"pytest, {len(list((PROJECT_ROOT / 'tests').glob('test_*.py')))} test files in <code>tests/</code> "
     "(including leakage tests)"],
    ["Reproducing", "<code>HOW_TO_RUN.md</code>; synthetic trial <code>scripts/try_synthetic.sh</code>; full run "
     "<code>scripts/run_full_scale.sh</code>"],
])}
<h3>Pipeline scripts (run in order)</h3>
{scripts_table()}
""")

    body = "".join(sec)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Project technical inventory</title>
<style>
@page {{ size: A4; margin: 16mm 14mm 16mm 14mm; }}
body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #1d1d1f;
       line-height: 1.4; background: #fff; }}
h1 {{ font-size: 20pt; margin: 0 0 2mm; color: #0b3d63; }}
h2 {{ font-size: 13.5pt; color: #0b3d63; border-bottom: 1.5px solid #0072B2; padding-bottom: 1mm;
      margin: 7mm 0 2.5mm; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 4mm 0 1.5mm; break-after: avoid; }}
p {{ margin: 1.5mm 0; }}
.sub {{ color: #555; margin-bottom: 4mm; }}
.note {{ background: #f3f6f9; border-left: 3px solid #0072B2; padding: 2mm 3mm; font-size: 8.5pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 8.4pt; }}
th, td {{ border: 0.6px solid #c9d1d9; padding: 1.2mm 1.8mm; vertical-align: top; text-align: left; }}
th {{ background: #e8f0f7; }}
tr {{ break-inside: avoid; }}
table.models td:nth-child(5) {{ font-size: 7.4pt; color: #333; word-break: break-word; }}
table.ref td:first-child {{ width: 32%; }}
code {{ font-family: Menlo, Consolas, monospace; font-size: 8pt; }}
a {{ color: #0b5cad; }}
p.mod {{ color: #444; font-style: italic; }}
.toc li {{ margin: 0.5mm 0; }}
</style></head><body>
<h1>Real-Time Session-Based E-Commerce Purchase Intention Prediction</h1>
<p class="sub">Technical inventory: dataset, methods, algorithms, code and tools · Karan Ray (deep learning),
Merve Kara (machine learning) · NCI MSc AI, Machine Learning · generated {date.today():%d %B %Y}</p>
<p class="note">This document is generated automatically by <code>scripts/23_project_inventory.py</code> from
<code>config/config.yaml</code>, the result tables and the code's docstrings. It is a technical reference to the
project, not report text.</p>
<ol class="toc"><li>Dataset</li><li>Features</li><li>Models and algorithms</li><li>Evaluation and statistics</li>
<li>Advanced methods and extensions</li><li>Headline results</li><li>Software, tools and deliverables</li>
<li>Appendix: module and function reference</li></ol>
{body}
<h2 style="break-before: page">Appendix: module and function reference</h2>
<p>Every public class and function in <code>src/</code> and <code>app/</code>, with the first line of its docstring.</p>
{module_reference()}
</body></html>"""


def chrome() -> str | None:
    for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium"):
        if Path(c).exists():
            return c
    return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    cfg = setup(p.parse_args(argv))
    OUT.mkdir(parents=True, exist_ok=True)
    with stage("Project inventory"):
        page = OUT / "project_inventory.html"
        page.write_text(build(cfg))
        exe = chrome()
        if exe:
            pdf = OUT / "project_inventory.pdf"
            subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True, timeout=120)
            print(f"-> {pdf.relative_to(PROJECT_ROOT)}")
        else:
            print("Chrome not found: open the HTML in a browser and print it to PDF.")
        print(f"-> {page.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
