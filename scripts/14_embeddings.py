"""Step 14: do learned product embeddings and LLM text embeddings improve the models?

Needs steps 1-2 for the mode (``data/interim/events_<mode>.parquet`` and ``data/processed``).
Embeddings use only events before the validation period (``split_cutoffs.json``):

* **item2vec**: PPMI-SVD product vectors from session co-occurrence (``--dim``);
* **text**: category path + brand embedded with ``nomic-embed-text`` via Ollama, PCA-reduced
  (skipped with a note if the embedding model is not pulled: ``ollama pull nomic-embed-text``).

For every K, LightGBM is trained with every project seed on four feature sets (base G1-G5,
+item2vec, +text, +both) and compared with the base set on the test weeks (paired bootstrap
on the primary seed, Holm-corrected per K).

Writes ``embeddings_ablation`` and ``embeddings_tests`` (CSV) to ``results/tables`` and caches
the vectors under ``data/interim/embeddings/``.

Examples::

    python scripts/14_embeddings.py --mode full --set data.session_sample_fraction=0.1
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import feature_list, processed_dir_for_k, resolve_path, set_seed  # noqa: E402
from src.data.sessionize import TARGET  # noqa: E402
from src.evaluation.statistics import holm, paired_bootstrap  # noqa: E402
from src.features.build_features import load_feature_table  # noqa: E402
from src.features.embeddings import item2vec, product_texts, session_features, text_vectors  # noqa: E402
from src.features.preprocessing import TabularPreprocessor  # noqa: E402
from src.llm.providers import OllamaEmbedder  # noqa: E402
from src.models.base import build_model  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402
from src.utils.notebook import prepared_data_status  # noqa: E402


def lgbm_inputs(cfg, table: pl.DataFrame, extra: list[str]):
    """Train/val/test matrices for LightGBM (raw profile) on G1-G5 plus ``extra`` numeric columns."""
    pre = TabularPreprocessor(features=feature_list(cfg) + extra, categorical=cfg["features"]["categorical"],
                              log1p=False, scale=False, categorical_encoding="ordinal",
                              top_n=cfg["features"]["top_n_categories"])
    parts = {s: table.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    pre.fit(parts["train"])
    out = {s: (pre.transform(d).to_numpy().astype(np.float32), d[TARGET].to_numpy().astype(np.int8))
           for s, d in parts.items()}
    return out, pre.output_columns


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--ks", type=int, nargs="+", help="prefix lengths (default: all prepared)")
    p.add_argument("--dim", type=int, default=16, help="item2vec dimensions")
    p.add_argument("--text-dim", type=int, default=8, help="text-embedding PCA dimensions")
    p.add_argument("--min-count", type=int, default=20, help="minimum events for a product to get a vector")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args(argv)
    cfg = setup(args)
    mode = cfg["data"]["mode"]
    ks = args.ks or [int(k) for k in cfg["sessionize"]["prefix_k"]]
    status = prepared_data_status(cfg)
    if any(status.get(k) != mode for k in ks):
        raise SystemExit(f"data/processed holds {status}, not {mode!r}; prepare it first.")
    store = ResultStore(cfg)
    seeds = [int(s) for s in cfg["project"]["seeds"]]
    cut = json.loads((Path(resolve_path(cfg, "processed_dir")) / "split_cutoffs.json").read_text())
    before = datetime.fromisoformat(cut["val_start"])
    events = pl.scan_parquet(Path(resolve_path(cfg, "interim_dir")) / f"events_{mode}.parquet")
    cache = Path(resolve_path(cfg, "interim_dir")) / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)

    with stage(f"item2vec ({args.dim} dims) from events before {before:%Y-%m-%d}"):
        f = cache / f"item2vec_{mode}_d{args.dim}_m{args.min_count}.parquet"
        i2v = pl.read_parquet(f) if f.exists() else item2vec(events, before, dim=args.dim, min_count=args.min_count,
                                                             seed=seeds[0])
        i2v.write_parquet(f)
        print(f"   {i2v.height:,} products with a vector")
    txt, note = None, None
    emb = OllamaEmbedder()
    ok, why = emb.available()
    if ok:
        with stage(f"Text embeddings ({emb.model} via Ollama, PCA {args.text_dim})"):
            f = cache / f"text_{mode}_{emb.model.replace(':', '_')}_d{args.text_dim}.parquet"
            if f.exists():
                txt = pl.read_parquet(f)
            else:
                txt = text_vectors(product_texts(events, before), emb, dim=args.text_dim, seed=seeds[0])
                txt.write_parquet(f)
            print(f"   {txt.height:,} category/brand texts embedded")
    else:
        note = f"text embeddings skipped: {why}"
        print(f"note: {note}")

    rows, tests = [], []
    for k in ks:
        with stage(f"K={k}: LightGBM with and without embeddings"):
            pdir = processed_dir_for_k(cfg, k)
            prefix = pl.read_parquet(pdir / "prefix_events.parquet",
                                     columns=["user_session", "pos", "product_id", "category_code", "brand"]
                                     ).with_columns(pl.col("category_code").fill_null("unknown"),
                                                    pl.col("brand").fill_null("unknown"))
            table = load_feature_table(cfg, k)
            sf = session_features(prefix, i2v, ["product_id"], "i2v_")
            table = table.join(sf, on="user_session", how="left")
            sets = {"base": [], "item2vec": [c for c in sf.columns if c != "user_session"]}
            if txt is not None:
                tf = session_features(prefix, txt, ["category_code", "brand"], "txt_")
                table = table.join(tf, on="user_session", how="left")
                sets["text"] = [c for c in tf.columns if c != "user_session"]
                sets["item2vec+text"] = sets["item2vec"] + sets["text"]
            primary: dict[str, np.ndarray] = {}
            for name, extra in sets.items():
                data, cols = lgbm_inputs(cfg, table, extra)
                for seed in seeds:
                    set_seed(seed)
                    m = build_model("lightgbm", cfg, seed=seed, columns=cols)
                    m.fit(data["train"][0], data["train"][1], data["val"][0], data["val"][1])
                    s = m.predict_proba(data["test"][0])
                    if seed == seeds[0]:
                        primary[name] = s
                    rows.append({"k": k, "feature_set": name, "seed": seed, "n_inputs": len(cols),
                                 "val_pr_auc": float(average_precision_score(data["val"][1], m.predict_proba(data["val"][0]))),
                                 "test_pr_auc": float(average_precision_score(data["test"][1], s))})
            y = data["test"][1]
            raw = []
            for name in sets:
                if name == "base":
                    continue
                t = paired_bootstrap(y, primary[name], primary["base"], "pr_auc", n_resamples=args.n_resamples, seed=seeds[0])
                raw.append({"k": k, "feature_set": name, "vs": "base", "diff": t["diff"], "lower": t["lower"],
                            "upper": t["upper"], "p_value": t["p_value"]})
            for r, ph in zip(raw, holm([r["p_value"] for r in raw])):
                r["p_holm"], r["significant"] = ph, ph < float(cfg["evaluation"]["alpha"])
            tests += raw
    res = pl.DataFrame(rows)
    res.write_csv(store.path("embeddings_ablation", "csv"))
    pl.DataFrame(tests).write_csv(store.path("embeddings_tests", "csv"))
    store.path("embeddings_summary", "json").write_text(json.dumps(
        {"mode": mode, "item2vec_products": i2v.height, "item2vec_dim": args.dim, "fitted_before": str(before),
         "text_model": emb.model if txt is not None else None, "note": note}, indent=2))
    print("\nTest PR-AUC (mean over seeds):")
    print_table(res.group_by("k", "feature_set").agg(pl.col("test_pr_auc").mean().round(4), pl.col("test_pr_auc").std().round(4).alias("sd"))
                .sort("k", "feature_set").to_dicts())
    print("\nVs the base features (primary seed):")
    print_table(pl.DataFrame(tests).select("k", "feature_set", pl.col("diff").round(4), pl.col("p_holm").round(3), "significant").to_dicts())


if __name__ == "__main__":
    main()
