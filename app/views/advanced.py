"""Advanced machine learning: Transformer, tuning, stacking, embeddings, conformal prediction, drift."""

from __future__ import annotations


import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
ink, pal = ui.INK[ui.theme()], ui.SERIES[ui.theme()]
ui.page_header("Advanced ML",
               "Six extensions beyond the main study, each tested on the same test weeks with paired "
               "bootstrap tests: does it really help, and by how much?", rq="Models")

POLICY = {"frozen": "Frozen (Oct–Dec)", "expanding": "Retrain on all past", "window_3": "Retrain on last 3 months",
          "window_1": "Retrain on last month"}


def verdict(sig: bool, diff: float) -> str:
    if sig:
        return ":green-badge[significant gain]" if diff > 0 else ":red-badge[significantly worse]"
    return ":gray-badge[no significant difference]"


tabs = st.tabs(["Transformer vs GRU", "Tuning (Optuna)", "Stacked ensembles", "Embeddings",
                "Conformal prediction", "Drift & retraining"])

# --------------------------------------------------------------------------- transformer
with tabs[0]:
    tr, tt, ta = s.csv("transformer"), s.csv("transformer_tests"), s.csv("transformer_attention")
    if tr is None:
        ui.missing("Transformer results", "16_transformer.py")
    else:
        st.markdown("A **Transformer encoder** reads the same clicks as the GRU, but every click can attend to "
                    "every other one directly. Same inputs, splits, training recipe and 3 seeds.")
        d = ui.per_k(tr)
        c = st.columns(max(d.height, 1))
        for col, r in zip(c, d.iter_rows(named=True)):
            col.metric(ui.label(r["model"]), f"{r['pr_auc_mean']:.4f}",
                       f"± {r['pr_auc_std']:.4f} · {r['n_parameters']:,} params · p95 {r['latency_p95_ms']:.2f} ms",
                       delta_color="off", delta_arrow="off")
        if tt is not None:
            for r in ui.per_k(tt).iter_rows(named=True):
                st.markdown(f"Transformer vs **{ui.label(r['vs'])}** after {K} clicks: {r['diff']:+.4f} PR-AUC "
                            f"(95% CI {r['lower']:+.4f} to {r['upper']:+.4f}, Holm p = {r['p_holm']:.2f}) "
                            + verdict(r["significant"], r["diff"]))
        if ta is not None:
            a = ui.per_k(ta)
            fig = go.Figure(go.Bar(x=[f"Click {p}" for p in a["position"]], y=a["attention"],
                                   marker=dict(color=pal[0], cornerradius=4),
                                   text=[f"{v:.0%}" for v in a["attention"]], textposition="outside", cliponaxis=False,
                                   textfont=dict(color=ink["text"], size=11)))
            ui.style(fig, height=300, legend=False)
            fig.update_yaxes(tickformat=".0%", title="Share of attention")
            st.subheader("Which clicks the Transformer looks at", anchor=False)
            ui.show(fig, key="attn")
            st.caption("Mean attention of the [CLS] token on each click, last layer, test visits. A rising profile "
                       "means the most recent clicks matter most.")

# --------------------------------------------------------------------------- tuning
with tabs[1]:
    tu, trials, imp = s.csv("tuning"), s.csv("tuning_trials"), s.csv("tuning_importance")
    if tu is None:
        ui.missing("tuning results", "15_tune.py")
    else:
        st.markdown("Bayesian optimisation (**Optuna, TPE sampler**). Every trial trains on the training weeks and "
                    "is scored on the later validation weeks; the best setting is retrained with 3 seeds and only "
                    "then compared on the test weeks with the RQ1 default.")
        for r in tu.iter_rows(named=True):
            st.markdown(f"**{ui.label(r['model'])}** (K={r['k']}, {r['trials']} trials): default "
                        f"{r['default_test_mean']:.4f} → tuned **{r['tuned_test_mean']:.4f}** "
                        f"(paired diff {r['diff']:+.4f}, Holm p = {r['p_holm']:.2f}) " + verdict(r["significant"], r["diff"]))
        left, right = st.columns([1.3, 1], gap="large")
        if trials is not None:
            with left:
                fig = go.Figure()
                for i, m in enumerate(trials["model"].unique(maintain_order=True)):
                    t = trials.filter(pl.col("model") == m).sort("trial")
                    fig.add_trace(go.Scatter(x=t["trial"], y=t["val_pr_auc"], mode="markers", name=ui.label(m),
                                             marker=dict(color=pal[i], size=8)))
                    fig.add_trace(go.Scatter(x=t["trial"], y=t["val_pr_auc"].cum_max(), mode="lines", showlegend=False,
                                             line=dict(color=pal[i], width=2)))
                ui.style(fig, height=320)
                fig.update_xaxes(title="Trial (0 = RQ1 default)")
                fig.update_yaxes(title="Validation PR-AUC")
                st.subheader("Search progress", anchor=False)
                ui.show(fig, key="trials")
        if imp is not None:
            with right:
                st.subheader("What mattered", anchor=False)
                st.dataframe(imp.sort("model", "importance", descending=[False, True]).with_columns(
                    pl.col("model").map_elements(ui.label, return_dtype=pl.String), pl.col("importance").round(3)),
                    hide_index=True, width="stretch")
                st.caption("fANOVA importance: share of the variance in validation PR-AUC explained by each "
                           "hyperparameter.")

# --------------------------------------------------------------------------- stacking
with tabs[2]:
    st_, stt, ov = s.csv("stacking"), s.csv("stacking_tests"), s.csv("error_overlap")
    if st_ is None:
        ui.missing("stacking results", "12_conformal_stacking.py")
    else:
        st.markdown("A logistic **meta-model** combines the models' scores. It is fitted on the validation weeks "
                    "(where every base model is out of sample) and judged on the test weeks against the best single "
                    "model, which is chosen on validation, not test.")
        t = ui.per_k(stt).sort("diff", descending=True) if stt is not None else None
        names = {"stack_ml": "ML models (Merve)", "stack_dl": "DL models (Karan)", "stack_all": "All six models",
                 "mean_logit_all": "Average of logits", "mean_rank_all": "Average of ranks"}
        if t is not None and t.height:
            fig = go.Figure(go.Bar(
                y=[names.get(x, x) for x in t["ensemble"]], x=t["diff"], orientation="h",
                error_x=dict(type="data", symmetric=False, array=(t["upper"] - t["diff"]).to_list(),
                             arrayminus=(t["diff"] - t["lower"]).to_list(), color=ink["muted"]),
                marker=dict(color=[ui.STATUS["good"] if sg else ink["muted"] for sg in t["significant"]], cornerradius=4),
                text=[f"{d:+.4f}" for d in t["diff"]], textposition="outside", cliponaxis=False,
                textfont=dict(color=ink["text"], size=11)))
            ui.style(fig, height=300, legend=False, left=170)
            fig.update_xaxes(title=f"PR-AUC gain over {ui.label(t['vs'][0])} (95% CI)", zeroline=True,
                             zerolinecolor=ink["muted"])
            fig.update_yaxes(autorange="reversed", showgrid=False)
            ui.show(fig, key="stack")
            st.caption("Green: significant after Holm correction.")
        if ov is not None:
            o = ui.per_k(ov).sort("missed_by_both", descending=True)
            with st.expander("Do the models miss the same buyers?"):
                st.dataframe(o.select(pl.col("model_a").map_elements(ui.label, return_dtype=pl.String).alias("Model A"),
                                      pl.col("model_b").map_elements(ui.label, return_dtype=pl.String).alias("Model B"),
                                      (pl.col("missed_by_both") * 100).round(1).alias("Missed buyers shared %"),
                                      pl.col("kappa").round(3).alias("Decision agreement (kappa)")),
                             hide_index=True, width="stretch")

# --------------------------------------------------------------------------- embeddings
with tabs[3]:
    ea, et = s.csv("embeddings_ablation"), s.csv("embeddings_tests")
    info = s.json("embeddings_summary") or {}
    if ea is None:
        ui.missing("embedding results", "14_embeddings.py")
    else:
        st.markdown(f"**item2vec**: {info.get('item2vec_products', 0):,} products embedded in "
                    f"{info.get('item2vec_dim', '?')} dimensions from which products appear together in sessions "
                    "(PPMI + SVD, events before the validation weeks only). **Text**: category and brand names "
                    "embedded by a language model (`nomic-embed-text`). Added to LightGBM's G1–G5 features.")
        if info.get("note"):
            st.caption(f":material/info: {info['note']}")
        m = ui.per_k(ea).group_by("feature_set").agg(pl.col("test_pr_auc").mean(), pl.col("test_pr_auc").std().alias("sd"))
        order = [x for x in ["base", "item2vec", "text", "item2vec+text"] if x in m["feature_set"].to_list()]
        m = m.with_columns(pl.col("feature_set").cast(pl.Enum(order))).sort("feature_set")
        fig = go.Figure(go.Bar(x=m["feature_set"].cast(pl.String), y=m["test_pr_auc"],
                               error_y=dict(type="data", array=m["sd"].fill_null(0).to_list(), color=ink["muted"]),
                               marker=dict(color=pal[0], cornerradius=4), text=[f"{v:.4f}" for v in m["test_pr_auc"]],
                               textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=11)))
        ui.style(fig, height=300, legend=False)
        lo = float(m["test_pr_auc"].min())
        fig.update_yaxes(range=[lo * 0.97, float(m["test_pr_auc"].max()) * 1.02], title="Test PR-AUC (LightGBM)")
        ui.show(fig, key="emb")
        if et is not None:
            for r in ui.per_k(et).iter_rows(named=True):
                st.markdown(f"+{r['feature_set']} vs base after {K} clicks: {r['diff']:+.4f} "
                            f"(Holm p = {r['p_holm']:.2f}) " + verdict(r["significant"], r["diff"]))

# --------------------------------------------------------------------------- conformal
with tabs[4]:
    cf, cw = s.csv("conformal"), s.csv("conformal_weekly")
    if cf is None:
        ui.missing("conformal results", "12_conformal_stacking.py")
    else:
        st.markdown("**Conformal prediction** turns a score into a *set* of labels with a guaranteed hit rate: the "
                    "true answer is in the set for at least 1 − α of visits. Calibrated per class on the validation "
                    "weeks (so buyers get the guarantee too), checked on the later test weeks.")
        c = ui.per_k(cf)
        models = ui.models_in(c)
        engine = c.filter(pl.col("engine"))["model"]
        model = st.selectbox("Model", models, index=models.index(engine[0]) if engine.len() else 0, key="cp_model",
                             format_func=ui.label)
        d = c.filter(pl.col("model") == model).sort("alpha")
        cols = st.columns(d.height)
        for col, r in zip(cols, d.iter_rows(named=True)):
            col.metric(f"Target {r['target_coverage']:.0%}", f"{r['coverage']:.1%} covered",
                       f"buyers {r['coverage_buyers']:.1%} · unsure {r['share_unsure']:.0%}",
                       delta_color="off", delta_arrow="off")
        fig = go.Figure()
        for i, (col, lab) in enumerate([("share_confident_buyer", "Confident: buyer"),
                                        ("share_unsure", "Unsure: both labels"),
                                        ("share_confident_non_buyer", "Confident: not a buyer")]):
            fig.add_trace(go.Bar(x=[f"{a:.0%} target" for a in d["target_coverage"]], y=d[col], name=lab,
                                 marker=dict(color=[pal[1], ink["muted"], pal[0]][i], cornerradius=2)))
        ui.style(fig, height=300, barmode="stack")
        fig.update_yaxes(tickformat=".0%", title="Share of test visits")
        ui.show(fig, key="cp")
        r10 = d.filter(pl.col("alpha") == 0.1)
        if r10.height:
            r = r10.row(0, named=True)
            st.markdown(f"At a 90 % guarantee the engine can **decide confidently for "
                        f"{1 - r['share_unsure'] - r['share_empty']:.0%} of visits**; of the visits it confidently "
                        f"calls buyers, **{r['precision_confident_buyer']:.0%} buy** (base rate "
                        f"{ui.per_k(s.csv('rq1_model_comparison'))['base_rate'][0]:.0%}). The rest are flagged *unsure* "
                        "instead of guessed.")
        if cw is not None:
            w = ui.per_k(cw)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["week"], y=w["coverage"], name="All visits", mode="lines+markers",
                                     line=dict(color=pal[0], width=2)))
            fig.add_trace(go.Scatter(x=w["week"], y=w["coverage_buyers"], name="Buyers", mode="lines+markers",
                                     line=dict(color=pal[1], width=2)))
            fig.add_hline(y=0.9, line=dict(color=ink["muted"], dash="dash"), annotation_text="90% target")
            ui.style(fig, height=280)
            fig.update_yaxes(tickformat=".0%", range=[0.8, 1.0])
            st.subheader(f"Does the guarantee hold week by week? ({ui.label(w['model'][0])})", anchor=False)
            ui.show(fig, key="cpw")

# --------------------------------------------------------------------------- drift
with tabs[5]:
    rt, rtt = s.csv("retraining"), s.csv("retraining_tests")
    if rt is None:
        ui.missing("retraining results", "13_drift_retraining.py")
    else:
        st.markdown("A deployment replay: each month from January 2020 is scored by LightGBM trained under four "
                    "policies (3 seeds each). Does a model trained on Oct–Dec decay, and does retraining help?")
        kk = ui.per_k(rt) if ui.per_k(rt).height else rt
        m = kk.group_by("month", "policy").agg(pl.col("pr_auc").mean(), pl.col("lift").mean(),
                                               pl.col("buyer_rate").first()).sort("month")
        metric = st.segmented_control("Show", ["pr_auc", "lift"], default="lift", key="rt_metric", required=True,
                                      format_func={"pr_auc": "PR-AUC", "lift": "Lift over random"}.get)
        fig = go.Figure()
        for i, p in enumerate(POLICY):
            d = m.filter(pl.col("policy") == p)
            fig.add_trace(go.Scatter(x=[ui.month_name(x) for x in d["month"]], y=d[metric], name=POLICY[p],
                                     mode="lines+markers",
                                     line=dict(color=pal[i], width=2, dash="dash" if p == "frozen" else "solid"),
                                     marker=dict(size=8)))
        ui.style(fig, height=320)
        fig.update_yaxes(title="PR-AUC" if metric == "pr_auc" else "Lift (PR-AUC / buyer rate)")
        fig.update_xaxes(type="category")
        ui.show(fig, key="rt")
        st.caption("PR-AUC also moves with each month's buyer rate; lift divides that out.")
        if rtt is not None:
            t = rtt.with_columns(pl.col("policy").replace_strict(POLICY, default=pl.col("policy")))
            st.dataframe(t.select("month", pl.col("policy").alias("Policy vs frozen"), pl.col("diff").round(4).alias("PR-AUC diff"),
                                  pl.col("p_holm").round(3).alias("Holm p"), pl.col("significant").alias("Significant")),
                         hide_index=True, width="stretch")
