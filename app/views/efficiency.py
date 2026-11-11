"""RQ4: accuracy vs latency, throughput, memory and size: is real-time scoring feasible?"""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
eff = s.csv("rq4_efficiency")
hyp = s.json("rq4_hypothesis")
hw = s.json("rq4_hardware")

ui.page_header("Fast enough for a live shop?",
               "Every saved model benchmarked in-process on one CPU thread: one session at a time and in batches.",
               rq="RQ4")
if eff is None:
    ui.missing("benchmarks", "05_benchmark_latency.py")
    st.stop()

colours = ui.model_colours()
ink = ui.INK[ui.theme()]
budget = float((hyp or {}).get("realtime_budget_ms_p95_single_session", 10.0))
part = ui.per_k(eff)
front = set(((hyp or {}).get("per_k", {}).get(str(K), {}) or {}).get("pareto_front_all_models", []))

c1, c2, c3 = st.columns(3)
within = (part["latency_b1_p95_ms"] <= budget).sum()
c1.metric("Within the real-time budget", f"{within} / {part.height}", f"p95 ≤ {budget:g} ms per session",
          delta_color="off")
fastest = part.filter(~pl.col("model").is_in(["rule_based"])).sort("latency_b1_p95_ms").row(0, named=True)
c2.metric("Fastest learned model", ui.label(fastest["model"]), f"{fastest['latency_b1_p95_ms']:.3f} ms p95",
          delta_color="off")
h4 = (hyp or {}).get("H4_trees_dominate_accuracy_latency_front")
with c3:
    if h4:
        st.markdown(f"<div style='padding:.6rem 0'><b>H4 · trees dominate the front</b><br>{ui.verdict_badge(h4['verdict'])}"
                    f"<div style='opacity:.65;font-size:.82rem;margin-top:.3rem'>Trees dominate at "
                    f"{h4['k_where_trees_dominate']} K</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader(f"Accuracy vs single-session latency · K={K}", anchor=False)
    d = part.sort("latency_b1_p95_ms")
    fig = go.Figure()
    fr = d.filter(pl.col("model").is_in(list(front))).sort("latency_b1_p95_ms")
    if fr.height > 1:
        fig.add_trace(go.Scatter(x=fr["latency_b1_p95_ms"], y=fr["test_pr_auc_primary"], mode="lines",
                                 line=dict(color=ink["muted"], width=1.5, dash="dot", shape="hv"),
                                 name="Pareto front", hoverinfo="skip"))
    front_order = [m for m in d["model"] if m in front]
    for r in d.iter_rows(named=True):
        m = r["model"]
        pos = ("top center" if front_order.index(m) % 2 == 0 else "bottom center") if m in front else "top center"
        fig.add_trace(go.Scatter(
            x=[r["latency_b1_p95_ms"]], y=[r["test_pr_auc_primary"]], mode="markers+text", name=ui.label(m),
            text=[ui.label(m) if m in front else ""], textposition=pos,
            textfont=dict(color=ink["text"], size=11),
            marker=dict(size=13, color=colours[m], symbol="circle" if m in front else "circle-open",
                        line=dict(color=colours[m] if m not in front else ink["surface"], width=2)),
            hovertemplate=(f"<b>{ui.label(m)}</b><br>p95 latency %{{x:.3f}} ms · PR-AUC %{{y:.4f}}<br>"
                           f"size {r['model_size_mb']:.2f} MB · {r['throughput_b4096_per_sec']:,.0f} sessions/s "
                           "(batch 4096)<extra></extra>"),
        ))
    fig.add_vline(x=budget, line=dict(color=ui.STATUS["critical"], width=1.5, dash="dash"),
                  annotation_text=f"budget {budget:g} ms", annotation_position="bottom left",
                  annotation_font=dict(color=ink["muted"], size=11))
    ui.style(fig, height=440, top=80)
    lo = float(np.log10(max(d["latency_b1_p95_ms"].min(), 1e-4) / 3))
    fig.update_xaxes(type="log", range=[lo, float(np.log10(budget * 2.5))],
                     title="p95 latency, batch of 1 (ms, log scale)")
    fig.update_yaxes(title="Test PR-AUC (primary seed)")
    ui.show(fig, key="pareto")
    st.caption("Filled, labelled markers are on the Pareto front (no other model is both faster and more accurate). "
               "Latency covers model inference only, not feature computation or network time.")

with right:
    st.subheader("Throughput", anchor=False)
    batch = st.segmented_control("Batch size", [1, 256, 4096], default=4096, key="batch", required=True)
    col = f"throughput_b{batch}_per_sec"
    d = part.sort(col)
    fig = go.Figure(go.Bar(
        y=[ui.label(m) for m in d["model"]], x=d[col], orientation="h",
        marker=dict(color=[colours[m] for m in d["model"]], cornerradius=4),
        text=[ui.fmt_int(v) for v in d[col]], textposition="outside", cliponaxis=False,
        textfont=dict(color=ink["text"], size=11),
        hovertemplate="%{y}: %{x:,.0f} sessions/s<extra></extra>",
    ))
    ui.style(fig, height=340, legend=False, left=150)
    fig.update_xaxes(type="log", title="Sessions per second (log scale)")
    fig.update_yaxes(showgrid=False)
    ui.show(fig, key="tput")

st.subheader("Cost sheet", anchor=False)
t = part.select(
    pl.col("model").replace_strict(ui.MODEL_LABELS, default=pl.col("model")).alias("Model"),
    pl.col("test_pr_auc_primary").alias("PR-AUC"),
    pl.col("latency_b1_p50_ms").alias("p50 ms"),
    pl.col("latency_b1_p95_ms").alias("p95 ms"),
    pl.col("latency_b1_p99_ms").alias("p99 ms"),
    pl.col("throughput_b4096_per_sec").alias("Sessions/s (4096)"),
    pl.col("model_size_mb").alias("Size MB"),
    pl.col("n_parameters").cast(pl.Float64, strict=False).alias("Parameters"),
    pl.col("train_time_sec_mean").alias("Train s"),
).sort("PR-AUC", descending=True)
st.dataframe(t, hide_index=True, width="stretch", column_config={
    "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
    "p50 ms": st.column_config.NumberColumn(format="%.3f"),
    "p95 ms": st.column_config.ProgressColumn(format="%.3f", min_value=0.0,
                                              max_value=float(max(budget, float(np.nanmax(part["latency_b1_p95_ms"]))))),
    "p99 ms": st.column_config.NumberColumn(format="%.3f"),
    "Sessions/s (4096)": st.column_config.NumberColumn(format="localized"),
    "Size MB": st.column_config.NumberColumn(format="%.3f"),
    "Parameters": st.column_config.NumberColumn(format="localized"),
    "Train s": st.column_config.NumberColumn(format="%.1f"),
})
if hw:
    st.caption(f"Hardware: {hw.get('platform')} · {hw.get('cpu_physical')} cores · {hw.get('ram_gb')} GB RAM · "
               f"Python {hw.get('python')} · benchmarks on 1 thread. Training times are indicative only.")
