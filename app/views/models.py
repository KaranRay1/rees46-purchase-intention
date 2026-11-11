"""RQ1 (model families) and RQ2 (how early: prefix length K)."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
rq1 = s.csv("rq1_model_comparison")
pairs = s.csv("rq1_pairwise_tests")
hyp = s.json("rq1_rq2_hypotheses")

ui.page_header("Which model family predicts best?",
               "Seven models, three prefix lengths, three seeds each, scored on the chronologically last 15 % of sessions.",
               rq="RQ1 · RQ2")
if rq1 is None:
    ui.missing("model comparison", "03_run_experiments.py")
    st.stop()

colours = ui.model_colours()
ink = ui.INK[ui.theme()]
METRICS = {
    "PR-AUC": ("pr_auc_mean", "Average precision on the test split (primary metric)"),
    "Lift ×": ("pr_auc_lift_mean", "PR-AUC ÷ base rate; comparable across K"),
    "ROC-AUC": ("roc_auc_mean", "Ranking quality across all thresholds"),
    "F1": ("f1_mean", "Positive-class F1 at the validation-tuned threshold"),
}

# --------------------------------------------------------------------------- RQ1 chart
top = st.columns([3, 1])
metric = top[1].selectbox("Metric", list(METRICS), help="All values are means over seeds.")
col, desc = METRICS[metric]
part = ui.per_k(rq1).sort(col)
base_rate = float(part["base_rate"][0])

fig = go.Figure()
for r in part.iter_rows(named=True):
    m = r["model"]
    err = None
    if col == "pr_auc_mean":
        err = dict(type="data", symmetric=False, array=[r["pr_auc_ci_high"] - r[col]],
                   arrayminus=[r[col] - r["pr_auc_ci_low"]], color=colours[m], thickness=2, width=0)
    fig.add_trace(go.Scatter(
        x=[r[col]], y=[ui.label(m)], mode="markers", name=ui.label(m), error_x=err,
        marker=dict(size=13, color=colours[m], line=dict(color=ink["surface"], width=2)),
        hovertemplate=(f"<b>{ui.label(m)}</b> · {ui.MODEL_FAMILY.get(m, '')}<br>"
                       f"PR-AUC {r['pr_auc_mean']:.4f} (95% CI {r['pr_auc_ci_low']:.4f}–{r['pr_auc_ci_high']:.4f})<br>"
                       f"Lift {r['pr_auc_lift_mean']:.2f}× · ROC-AUC {r['roc_auc_mean']:.4f}<br>"
                       f"F1 {r['f1_mean']:.4f} · precision {r['precision_mean']:.3f} · recall {r['recall_mean']:.3f}"
                       "<extra></extra>"),
    ))
ui.style(fig, height=360, legend=False, left=150)
fig.update_xaxes(title=f"{metric} at K={K}" + (f"   ·   random ranking = base rate {base_rate:.1%}"
                                                  if col == "pr_auc_mean" else ""))
fig.update_yaxes(showgrid=False)
top[0].subheader(f"{metric} by model · K={K}", anchor=False)
with top[0]:
    ui.show(fig, key="rq1")
top[0].caption(desc + (". Bars show the bootstrap 95 % CI (primary seed)." if col == "pr_auc_mean" else "."))

with top[1]:
    st.write("")
    if hyp:
        for key, title in (("H1a_trees_beat_logistic_regression", "H1a · trees > LR"),
                           ("H1b_deep_tabular_beat_logistic_regression", "H1b · deep tabular > LR"),
                           ("H1c_gru_matches_best_aggregated_at_max_k", "H1c · GRU ≈ best at max K")):
            h = hyp.get(key)
            if h:
                st.markdown(f"<div style='margin:.5rem 0 .15rem 0;font-weight:600'>{title}</div>"
                            f"{ui.verdict_badge(h['verdict'])}", unsafe_allow_html=True)
                if "significant_wins" in h:
                    st.caption(f"Significant wins after Holm correction: {h['significant_wins']}")
                elif "diff" in h:
                    st.caption(f"GRU − {ui.label(h['best_aggregated_model'])}: {h['diff']:+.4f} PR-AUC")

st.divider()

# --------------------------------------------------------------------------- RQ2 chart
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("Does seeing more events help?", anchor=False)
    ks = sorted(set(rq1["k"].to_list()))
    fig = go.Figure()
    for m in ui.models_in(rq1):
        d = rq1.filter(pl.col("model") == m).sort("k")
        fig.add_trace(go.Scatter(
            x=[f"K={x}" for x in d["k"]], y=d["pr_auc_lift_mean"], name=ui.label(m), mode="lines+markers",
            line=dict(color=colours[m], width=2), marker=dict(size=8, line=dict(color=ink["surface"], width=1.5)),
            customdata=np.stack([d["pr_auc_mean"], d["base_rate"]], axis=-1),
            hovertemplate=f"<b>{ui.label(m)}</b><br>%{{x}}: lift %{{y:.2f}}× · PR-AUC %{{customdata[0]:.4f}}"
                          "<br>base rate %{customdata[1]:.2%}<extra></extra>",
        ))
    ui.style(fig, height=420, top=90, hovermode="x unified")
    fig.update_yaxes(title="PR-AUC lift over base rate (×)")
    ui.show(fig, key="rq2")
with right:
    st.write("")
    h2 = (hyp or {}).get("H2_performance_rises_with_k_diminishing")
    if h2:
        st.markdown(f"**H2 · performance rises with K** {ui.verdict_badge(h2['verdict'])}", unsafe_allow_html=True)
        st.caption(f"Models rising with K: {h2['models_rising']} · with diminishing gains: {h2['models_diminishing']}")
    st.markdown(
        "Lift is measured against each K's own base rate, because the population changes with K: "
        "a session must have **more than K events** and **no purchase in its first K** to be eligible, "
        "so larger K means fewer, longer, harder sessions."
    )
    summary = s.json("data_summary")
    if summary:
        rows = [{"K": int(k_), "Eligible sessions": v["eligible"], "Buy later": f"{v['positive_rate']:.2%}"}
                for k_, v in sorted(summary["sessionize"].items(), key=lambda kv: int(kv[0]))]
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"Eligible sessions": st.column_config.NumberColumn(format="localized")})

st.divider()

# --------------------------------------------------------------------------- pairwise
st.subheader(f"Head-to-head at K={K}", anchor=False)
if pairs is None:
    ui.missing("pairwise tests", "03_run_experiments.py")
else:
    p = ui.per_k(pairs)
    models = [m for m in ui.model_order() if m in set(p["model_a"]) | set(p["model_b"])]
    n = len(models)
    diff = np.full((n, n), np.nan)
    text = [["" for _ in range(n)] for _ in range(n)]
    hover = [["" for _ in range(n)] for _ in range(n)]
    for r in p.iter_rows(named=True):
        i, j = models.index(r["model_a"]), models.index(r["model_b"])
        sig = r["p_bootstrap_holm"] < 0.05
        for a, b, sign in ((i, j, 1), (j, i, -1)):
            d = sign * r["pr_auc_diff"]
            diff[a, b] = d
            text[a][b] = f"{d:+.3f}" + ("" if sig else "·ns")
            hover[a][b] = (f"<b>{ui.label(models[a])}</b> − <b>{ui.label(models[b])}</b><br>ΔPR-AUC {d:+.4f}<br>"
                           f"Holm p (bootstrap) {r['p_bootstrap_holm']:.3g} · {r['family']}")
    lim = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    neutral = ui.INK[ui.theme()]["neutral"]
    fig = go.Figure(go.Heatmap(
        z=diff, x=[ui.label(m) for m in models], y=[ui.label(m) for m in models], text=text,
        texttemplate="%{text}", textfont=dict(size=11), hovertext=hover, hoverinfo="text",
        colorscale=[[0, ui.DIVERGING["neg"]], [0.5, neutral], [1, ui.DIVERGING["pos"]]], zmin=-lim, zmax=lim,
        xgap=2, ygap=2, colorbar=dict(title="ΔPR-AUC", thickness=10, outlinewidth=0),
    ))
    ui.style(fig, height=470, legend=False, left=150, top=70, bottom=16)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    ui.show(fig, key="pairs")
    st.caption("Row minus column test PR-AUC (primary seed). Red: the row model is better. "
               "`ns`: not significant after Holm correction (paired bootstrap, α = 0.05).")

with st.expander("Full results table"):
    cols = ["k", "model", "n_seeds", "base_rate", "pr_auc_mean", "pr_auc_std", "pr_auc_ci_low", "pr_auc_ci_high",
            "pr_auc_lift_mean", "roc_auc_mean", "f1_mean", "precision_mean", "recall_mean", "train_time_sec_mean"]
    t = rq1.select([c for c in cols if c in rq1.columns]).with_columns(pl.col("model").replace_strict(ui.MODEL_LABELS, default=pl.col("model")))
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in cols[3:]})
    st.caption("Training times are indicative only: some runs shared the machine with other jobs.")
