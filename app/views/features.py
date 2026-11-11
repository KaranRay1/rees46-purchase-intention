"""RQ3: what each feature group adds (ablation ladder) and the ratios-instead-of-counts variant."""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
abl = s.csv("rq3_ablation")
tests = s.csv("rq3_step_tests")
hyp = s.json("rq3_hypothesis")

ui.page_header("Which behaviours carry the signal?",
               "Feature groups are added one at a time (G1 → G5) and each model is retrained on every rung.",
               rq="RQ3")
if abl is None:
    ui.missing("ablation results", "04_run_ablation.py")
    st.stop()

colours = ui.model_colours()
ink = ui.INK[ui.theme()]
part = ui.per_k(abl)
ladder = part.filter(~pl.col("is_variant")).sort("step")
models = ui.models_in(part)
step_names = {r["step"]: r["feature_set"].split(":", 1)[-1] for r in ladder.iter_rows(named=True)}
added = {1: "G1 Activity", 2: "+ G2 Cart", 3: "+ G3 Ratio", 4: "+ G4 Category", 5: "+ G5 Price & time"}

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader(f"PR-AUC up the ladder · K={K}", anchor=False)
    fig = go.Figure()
    for m in models:
        d = ladder.filter(pl.col("model") == m)
        fig.add_trace(go.Scatter(
            x=[added.get(x, f"S{x}") for x in d["step"]], y=d["pr_auc_mean"], name=ui.label(m),
            mode="lines+markers+text", line=dict(color=colours[m], width=2, shape="linear"),
            marker=dict(size=9, line=dict(color=ink["surface"], width=1.5)),
            text=[f"{v:.3f}" if i == d.height - 1 else "" for i, v in enumerate(d["pr_auc_mean"])],
            textposition="top center" if m == models[0] else "bottom center",
            textfont=dict(color=ink["muted"], size=11),
            customdata=d.select("feature_set", "n_inputs", "pr_auc_gain").rows(),
            hovertemplate=f"<b>{ui.label(m)}</b> · %{{customdata[0]}}<br>PR-AUC %{{y:.4f}} "
                          "(gain %{customdata[2]:+.4f})<br>%{customdata[1]} model inputs<extra></extra>",
        ))
    ui.style(fig, height=420, top=70, hovermode="x unified")
    fig.update_yaxes(title="Test PR-AUC (mean of seeds)")
    ui.show(fig, key="ladder")

with right:
    st.subheader("Gain from each step", anchor=False)
    focus = st.segmented_control("Model", models, default=models[0], format_func=ui.label, key="abl_model",
                                 required=True)
    t = (tests.filter((pl.col("k") == K) & (pl.col("model") == focus) & (pl.col("family") != "variant")).sort("step")
         if tests is not None else None)
    if t is not None and t.height:
        sig = t["p_bootstrap_holm"] < 0.05
        fig = go.Figure(go.Bar(
            y=[added.get(x, x) for x in t["step"]], x=t["pr_auc_gain"], orientation="h",
            marker=dict(color=[colours[focus] if s_ else ink["grid"] for s_ in sig], cornerradius=4),
            error_x=dict(type="data", symmetric=False, array=(t["gain_ci_high"] - t["pr_auc_gain"]).to_list(),
                         arrayminus=(t["pr_auc_gain"] - t["gain_ci_low"]).to_list(), color=ink["muted"],
                         thickness=1.5, width=3),
            text=[f"{g:+.4f}" for g in t["pr_auc_gain"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            customdata=t.select("p_bootstrap_holm", "family").rows(),
            hovertemplate="%{y}: %{x:+.4f} PR-AUC<br>Holm p %{customdata[0]:.3g} · %{customdata[1]}<extra></extra>",
        ))
        ui.style(fig, height=300, legend=False, left=140)
        hi = float(t["gain_ci_high"].max())
        fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
        fig.update_xaxes(title="PR-AUC gain vs previous step", range=[min(0.0, float(t["gain_ci_low"].min())) - hi * 0.08, hi * 1.35])
        ui.show(fig, key="gains")
        st.caption("Coloured: significant after Holm correction. Grey: not significant. Whiskers: bootstrap 95 % CI.")
    if hyp:
        h = hyp.get("H3_behavioural_ratios_add_pr_auc")
        if h:
            st.markdown(f"**H3 · ratios add PR-AUC** {ui.verdict_badge(h['verdict'])}", unsafe_allow_html=True)
            st.caption(f"Significant gains: {h['significant_gains']}. With no remove events and fixed K, the "
                       "cart/view ratio is a function of cart_count, which G2 already contains.")

st.divider()
st.subheader("Can the ratio replace the cart block?", anchor=False)
var = part.filter(pl.col("is_variant"))
if var.height:
    rows = []
    for m in models:
        base = ladder.filter((pl.col("model") == m) & (pl.col("step") == 1))["pr_auc_mean"]
        full = ladder.filter((pl.col("model") == m) & (pl.col("step") == 2))["pr_auc_mean"]
        v = var.filter(pl.col("model") == m)["pr_auc_mean"]
        if base.len() and full.len() and v.len():
            rows.append((m, base[0], v[0], full[0]))
    fig = go.Figure()
    labels = ["G1 only", "G1 + ratio (variant)", "G1 + G2 cart block"]
    for i, lab in enumerate(labels):
        fig.add_trace(go.Bar(
            name=lab, x=[ui.label(r[0]) for r in rows], y=[r[i + 1] for r in rows],
            marker=dict(color=ui.SEQUENTIAL_BLUE[2 + 2 * i], cornerradius=4, line=dict(color=ink["surface"], width=2)),
            text=[f"{r[i + 1]:.3f}" for r in rows], textposition="outside", textfont=dict(color=ink["text"], size=11),
            hovertemplate=f"{lab}<br>%{{x}}: PR-AUC %{{y:.4f}}<extra></extra>",
        ))
    ui.style(fig, height=340, top=60, barmode="group")
    fig.update_yaxes(title="Test PR-AUC", rangemode="tozero")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        ui.show(fig, key="variant")
    c2.markdown("A single cart/view ratio recovers most of the cart signal, but the full G2 block is still "
                "significantly better, because it also knows **whether**, **when** (first cart position) and "
                "**most recently** (last event type) the shopper carted.")

with st.expander("Feature groups"):
    groups = ui.cfg()["features"]["groups"]
    st.dataframe([{"Group": ui.GROUP_LABELS.get(g, g), "Features": ", ".join(f)} for g, f in groups.items()],
                 hide_index=True, width="stretch")
with st.expander("All step tests"):
    if tests is not None:
        st.dataframe(ui.per_k(tests), hide_index=True, width="stretch")
