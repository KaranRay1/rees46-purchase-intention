"""Overview: the question, headline numbers, the hypothesis scoreboard and the pipeline."""

import html

import polars as pl
import streamlit as st

import ui

s, K = ui.store(), ui.k()
summary = s.json("data_summary")
rq1 = s.csv("rq1_model_comparison")
rq4 = s.csv("rq4_efficiency")
board = s.csv("hypothesis_scoreboard")

ui.page_header(
    "Will this shopper buy?",
    "Predicting a purchase from the first few clicks of an e-commerce session: a comparison of "
    "tabular machine learning and deep neural models on the REES46 event log.",
)

# --------------------------------------------------------------------------- headline numbers
c1, c2, c3, c4 = st.columns(4)
if summary:
    clean = summary.get("cleaning")  # absent when step 01 ran with --no-report
    if clean:
        c1.metric("Sessions analysed", ui.fmt_int(clean["sessions_out"]),
                  help=f"{clean['rows_out']:,} events after cleaning, from {', '.join(clean['sources'])}")
    sk = summary.get("sessionize", {}).get(str(K))
    if sk:
        c2.metric(f"Eligible at K={K}", ui.fmt_int(sk["eligible"]),
                  f"{sk['positive_rate']:.1%} buy later", delta_color="off",
                  help="Sessions with more than K events and no purchase within the first K.")
if rq1 is not None:
    part = ui.per_k(rq1)
    best = part.sort("pr_auc_mean", descending=True).row(0, named=True)
    rule = part.filter(pl.col("model") == "rule_based")
    c3.metric(f"Best PR-AUC at K={K}", f"{best['pr_auc_mean']:.3f}", f"{ui.label(best['model'])}",
              delta_color="off", help="Test-set average precision, mean of 3 seeds.")
    gain = best["pr_auc_mean"] - rule["pr_auc_mean"][0] if rule.height else None
    c4.metric("Lift over base rate", f"{best['pr_auc_lift_mean']:.2f}×",
              f"+{gain:.3f} PR-AUC vs rule" if gain is not None else None, delta_color="normal",
              help="PR-AUC divided by the test base rate: how much better than random ranking.")
if rq4 is not None and rq1 is not None:
    e = ui.per_k(rq4).filter(pl.col("model") == best["model"])
    if e.height:
        st.caption(f":material/speed: {ui.label(best['model'])} scores one session in "
                   f"**{e['latency_b1_p95_ms'][0]:.2f} ms** (p95, one CPU thread), well inside a 10 ms real-time budget.")

st.write("")

# --------------------------------------------------------------------------- scoreboard
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Hypothesis scoreboard", anchor=False)
    if board is None:
        ui.missing("hypothesis scoreboard", "03_run_experiments.py")
        st.caption("The scoreboard is written by notebook 03 (and completed by notebook 04).")
    else:
        rows = "".join(
            f"<tr><td style='font-weight:700;white-space:nowrap;padding:.55rem .6rem .55rem 0'>{html.escape(r['id'])}</td>"
            f"<td style='padding:.55rem .6rem'>{html.escape(r['hypothesis'])}"
            f"<div style='opacity:.6;font-size:.8rem'>{html.escape(str(r['evidence']))}</div></td>"
            f"<td style='padding:.55rem 0;text-align:right'>{ui.verdict_badge(r['verdict'])}</td></tr>"
            for r in board.iter_rows(named=True)
        )
        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:.92rem'>"
            "<tbody style='border-top:1px solid rgba(128,128,128,.25)'>"
            + rows.replace("<tr>", "<tr style='border-bottom:1px solid rgba(128,128,128,.18)'>")
            + "</tbody></table>",
            unsafe_allow_html=True,
        )

with right:
    st.subheader("What the data says", anchor=False)
    if rq1 is None:
        ui.missing("model results", "03_run_experiments.py")
    else:
        part = ui.per_k(rq1).sort("pr_auc_mean", descending=True)
        top3 = part.head(3)
        spread = top3["pr_auc_mean"].max() - top3["pr_auc_mean"].min()
        lr = part.filter(pl.col("model") == "logistic_regression")
        points = [
            f"**Top three are a near tie at K={K}:** {', '.join(ui.label(m) for m in top3['model'])} "
            f"are within {spread:.3f} PR-AUC of each other.",
        ]
        if lr.height:
            points.append(f"**Non-linear beats linear:** the best model is "
                          f"+{part['pr_auc_mean'][0] - lr['pr_auc_mean'][0]:.3f} PR-AUC above logistic regression.")
        abl = s.csv("rq3_ablation")
        if abl is not None:
            a = ui.per_k(abl).filter((pl.col("model") == "lightgbm") & ~pl.col("is_variant")).sort("step")
            if a.height >= 2:
                points.append(f"**Cart behaviour is the signal:** adding the cart group lifts LightGBM from "
                              f"{a['pr_auc_mean'][0]:.3f} to {a['pr_auc_mean'][1]:.3f} PR-AUC.")
        ex = s.json("rq5_explain_summary")
        if ex and ex.get("buyers_missed_by_all"):
            m = ex["buyers_missed_by_all"]
            points.append(f"**A hard ceiling:** {m['n'] / m['of']:.0%} of test buyers are missed by every model. "
                          "None of them carted in the prefix.")
        for p in points:
            st.markdown(f"- {p}")

st.write("")

# --------------------------------------------------------------------------- pipeline
st.subheader("How the study runs", anchor=False)
steps = [
    ("01", "Prepare", "Clean events, cut each session at K events, chronological 70/15/15 split."),
    ("02", "Features", "18 prefix features in 5 groups + raw event sequences for the GRU."),
    ("03", "Models", "7 models × 3 K × 3 seeds; bootstrap and McNemar tests (RQ1–2)."),
    ("04", "Ablation", "Feature groups added one at a time on LightGBM and MLP (RQ3)."),
    ("05", "Benchmark", "Latency, throughput, memory and size on one CPU thread (RQ4)."),
    ("06", "Explain", "SHAP / attention by feature group, error analysis (RQ5)."),
]
cols = st.columns(len(steps))
for col, (n, title, text) in zip(cols, steps):
    col.markdown(f"<div class='card'><div class='step'>STEP {n}</div><h4>{title}</h4><p>{text}</p></div>",
                 unsafe_allow_html=True)
st.caption("Each step is a script in `scripts/` (`python scripts/0N_….py --mode dev`); this dashboard only reads "
           "what they wrote.")
