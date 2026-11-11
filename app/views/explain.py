"""RQ5: what drives predictions, do model families agree, and who gets missed."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
groups = s.csv(f"rq5_importance_groups_k{K}")
feats = s.csv(f"rq5_importance_features_k{K}")
agree = s.csv(f"rq5_rank_agreement_k{K}")
errors = s.csv(f"rq5_error_profile_k{K}")
summary = s.json("rq5_explain_summary")
hyp = s.json("rq5_hypothesis")

ui.page_header("Why does a model say “buyer”?",
               "SHAP attributions (TreeSHAP, LinearSHAP, DeepSHAP) and TabNet attention on a test sample, "
               "summed into the five feature groups.", rq="RQ5")
if groups is None:
    ui.missing(f"explanations for K={K}", "06_explain.py")
    st.stop()

ink = ui.INK[ui.theme()]
gcol = ui.group_colours()
models = [m for m in ui.model_order() if m in groups.columns]

# --------------------------------------------------------------------------- group shares
left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader(f"Share of attribution by feature group · K={K}", anchor=False)
    fig = go.Figure()
    for g in ui.GROUP_LABELS:
        row = groups.filter(pl.col("feature") == g)
        if not row.height:
            continue
        vals = [float(row[m][0]) for m in models]
        fig.add_trace(go.Bar(
            name=ui.GROUP_LABELS[g], y=[ui.label(m) for m in models], x=vals, orientation="h",
            marker=dict(color=gcol[g], line=dict(color=ink["surface"], width=2)),
            text=[f"{v:.0%}" if v >= 0.08 else "" for v in vals], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate=f"<b>%{{y}}</b><br>{ui.GROUP_LABELS[g]}: %{{x:.1%}}<extra></extra>",
        ))
    ui.style(fig, height=380, left=150, top=70, barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of mean |attribution|")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    ui.show(fig, key="shares")
    st.caption("TabNet uses its attention masks, not SHAP, so its row is comparable only qualitatively.")
with right:
    st.write("")
    h = (hyp or {}).get("H5_cart_signals_rank_highest_for_every_family")
    if h:
        st.markdown(f"**H5 · cart signals rank first** {ui.verdict_badge(h['verdict'])}", unsafe_allow_html=True)
        st.caption(f"Cart group ranked first in {h['cart_group_top']} (model, K) pairs.")
    st.markdown(
        "**Use vs value.** Price & time receives the largest share, yet the ablation shows adding it gains "
        "only ~0.015 PR-AUC, while the cart group adds 0.07–0.13. Attribution measures how much a model "
        "*leans on* a group (and grows with the group's size: G5 has six features), not how much the "
        "group *adds*. Read this page together with RQ3."
    )

st.divider()

# --------------------------------------------------------------------------- features + agreement
c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    st.subheader("Top features", anchor=False)
    if feats is not None:
        options = ["mean_share", *models]
        who = st.selectbox("Model", options, format_func=lambda m: "Mean over models" if m == "mean_share" else ui.label(m))
        group_of = {f: g for g, fs in ui.cfg()["features"]["groups"].items() for f in fs}
        top = feats.sort(who, descending=True).head(10).reverse()
        fig = go.Figure(go.Bar(
            y=top["feature"], x=top[who], orientation="h",
            marker=dict(color=[gcol.get(group_of.get(f, ""), ink["muted"]) for f in top["feature"]], cornerradius=4),
            text=[f"{v:.1%}" for v in top[who]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            customdata=[ui.GROUP_LABELS.get(group_of.get(f, ""), "") for f in top["feature"]],
            hovertemplate="<b>%{y}</b> · %{customdata}<br>%{x:.1%} of attribution<extra></extra>",
        ))
        ui.style(fig, height=380, legend=False, left=190)
        fig.update_xaxes(tickformat=".0%", title="Share of mean |attribution|")
        fig.update_yaxes(showgrid=False)
        ui.show(fig, key="topfeat")
        ui.legend_items((ui.GROUP_LABELS[g], gcol[g]) for g in ui.GROUP_LABELS)
with c2:
    st.subheader("Do models agree?", anchor=False)
    if agree is not None:
        level = st.segmented_control("Level", ["group", "feature"], default="group", key="agree_level",
                                     required=True)
        a = agree.filter(pl.col("level") == level)
        ms = [m for m in models if m in set(a["model_a"]) | set(a["model_b"])]
        z = np.eye(len(ms))
        for r in a.iter_rows(named=True):
            i, j = ms.index(r["model_a"]), ms.index(r["model_b"])
            z[i, j] = z[j, i] = r["spearman_rho"]
        n = len(ui.SEQUENTIAL_BLUE) - 1
        fig = go.Figure(go.Heatmap(
            z=z, x=[ui.label(m) for m in ms], y=[ui.label(m) for m in ms], zmin=-1 if z.min() < 0 else 0, zmax=1,
            colorscale=[[i / n, c] for i, c in enumerate(ui.SEQUENTIAL_BLUE)],
            text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
            hovertemplate="%{y} vs %{x}: ρ = %{z:.2f}<extra></extra>",
            colorbar=dict(title="ρ", thickness=10, outlinewidth=0),
        ))
        ui.style(fig, height=400, legend=False, left=150, bottom=90)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, tickangle=-30)
        ui.show(fig, key="agree")
        st.caption("Spearman rank correlation of importance rankings.")

st.divider()

# --------------------------------------------------------------------------- errors
st.subheader("Who gets missed?", anchor=False)
if summary and summary.get("buyers_missed_by_all"):
    m = summary["buyers_missed_by_all"]
    prof = pl.DataFrame(summary["missed_by_all_profile"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Buyers missed by every model", f"{m['n'] / m['of']:.0%}", f"{m['n']:,} of {m['of']:,} test buyers",
              delta_color="off", help="At each model's tuned threshold, including the GRU and the rule baseline "
                                      "(K=5, from notebook 04).")
    caught = prof.filter(pl.col("status").str.contains("caught"))
    missed = prof.filter(pl.col("status") == "buyer missed by all")
    if caught.height and missed.height:
        c2.metric("…of whom carted in the prefix", f"{missed['has_cart'][0]:.0%}",
                  f"vs {caught['has_cart'][0]:.0%} of buyers caught", delta_color="off")
        c3.metric("Distinct products viewed", f"{missed['distinct_products'][0]:.1f}",
                  f"caught buyers: {caught['distinct_products'][0]:.1f}", delta_color="off")
    st.dataframe(prof, hide_index=True, width="stretch")
    st.caption("A missed buyer looks like a non-buyer in the first K events: no cart, similar browsing. This is a "
               "limit of what a short prefix contains, not of any one model family.")
if errors is not None:
    with st.expander(f"Error profile per model · K={K}"):
        st.dataframe(errors.with_columns(pl.col("model").replace_strict(ui.MODEL_LABELS, default=pl.col("model"))),
                     hide_index=True, width="stretch")

figs = [p for p in s.figures("rq5_waterfall_lightgbm") if f"_k{K}_" in p.name]
if figs:
    with st.expander(f"Individual explanations (LightGBM waterfalls, K={K})"):
        kind = st.segmented_control("Outcome", ["true_positive", "false_positive", "false_negative", "true_negative"],
                                    default="false_negative", format_func=lambda x: x.replace("_", " "),
                                    key="wf_kind", required=True)
        sel = [p for p in figs if kind in p.name]
        for col, p in zip(st.columns(max(len(sel), 1)), sel):
            col.image(str(p), width="stretch")
