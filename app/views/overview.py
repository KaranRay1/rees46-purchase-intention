"""Overview: the problem, the research → analysis → product journey, the two teams, and what we are sure of."""

import polars as pl
import streamlit as st

import team_page
import ui

s, K = ui.store(), ui.k()
summary = s.json("data_summary")
rq1 = s.csv("rq1_model_comparison")
rq4 = s.csv("rq4_efficiency")
pairs = s.csv("rq1_pairwise_tests")
abl = s.csv("rq3_ablation")
ex = s.json("rq5_explain_summary")

ui.page_header(
    "Will this shopper buy?",
    "A purchase-intent engine that reads the first few clicks of an online-shop visit and estimates, in "
    "milliseconds, whether the visitor will go on to buy, so the shop can act while the visit is still live.",
)

# --------------------------------------------------------------------------- journey
cols = st.columns(3)
journey = [
    ("1 · RESEARCH", "Question and data",
     "Literature on session-based purchase prediction, and a public log of real shop clicks (REES46) as evidence."),
    ("2 · ANALYSIS", "Two teams, seven models",
     "Merve: classic machine learning. Karan: deep learning. Same data, same fair test on a later time period."),
    ("3 · PRODUCT", "A real-time engine",
     "The trained models behind one scoring function: clicks in, purchase probability and next action out."),
]
for col, (kick, title, body) in zip(cols, journey):
    col.markdown(ui.card(title, body, kick), unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------- headline numbers
c1, c2, c3, c4 = st.columns(4)
clean = (summary or {}).get("cleaning")
if clean:
    c1.metric("Shop visits analysed", ui.fmt_int(clean["sessions_out"]),
              f"{ui.fmt_int(clean['rows_out'])} clicks", delta_color="off", delta_arrow="off")
if rq1 is not None:
    part = ui.per_k(rq1)
    top = part.sort("pr_auc_mean", descending=True).row(0, named=True)
    c2.metric(f"Best PR-AUC after {K} clicks", f"{top['pr_auc_mean']:.3f}",
              f"{ui.label(top['model'])} · {ui.TEAMS[ui.team_of(top['model'])]['title']}", delta_color="off",
              delta_arrow="off")
    c3.metric("Better than guessing", f"{top['pr_auc_lift_mean']:.1f}×",
              f"PR-AUC {top['pr_auc_mean']:.3f} vs {top['base_rate']:.3f} random", delta_color="off", delta_arrow="off",
              help="PR-AUC divided by the share of visitors who buy: how much better than random the ranking is.")
    if rq4 is not None:
        e = ui.per_k(rq4).filter(pl.col("model") == top["model"])
        if e.height:
            c4.metric("Time to score one visit", f"{e['latency_b1_p95_ms'][0]:.2f} ms",
                      "budget for real time: 10 ms", delta_color="off", delta_arrow="off",
                      help="95th percentile, one CPU thread, model inference only.")

st.write("")

# --------------------------------------------------------------------------- teams
st.subheader("Who built what", anchor=False)
cols = st.columns(2, gap="large")
for col, team in zip(cols, ("ml", "dl")):
    t = ui.TEAMS[team]
    with col:
        ui.author_chip(team)
        names = ", ".join(ui.label(m) for m in t["models"])
        line = ""
        if rq1 is not None:
            d = ui.per_k(rq1).filter(pl.col("model").is_in(t["models"])).sort("pr_auc_mean", descending=True)
            if d.height:
                line = f" Best: **{ui.label(d['model'][0])}**, PR-AUC {d['pr_auc_mean'][0]:.3f}."
        st.markdown(f"{t['pitch']}  \n{names}.{line}")

st.write("")

# --------------------------------------------------------------------------- confident findings
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("What the evidence supports", anchor=False)
    points = []
    if rq1 is not None:
        wins = [m for m in ("random_forest", "lightgbm", "mlp", "tabnet")
                if (r := team_page.significant_vs(pairs, K, m, "logistic_regression")) and r[1] and r[0] > 0]
        if wins:
            points.append(f"**Non-linear models win.** {', '.join(ui.label(m) for m in wins)} all beat logistic "
                          "regression, and the gap is statistically significant.")
        part = ui.per_k(rq1).sort("pr_auc_mean", descending=True)
        spread = float(part["pr_auc_mean"][0] - part.filter(pl.col("model") != "rule_based")["pr_auc_mean"].sort()[-3])
        points.append(f"**Machine learning and deep learning are close.** The top three models are within "
                      f"{spread:.3f} PR-AUC; neither family is clearly better.")
    if abl is not None:
        a = ui.per_k(abl).filter((pl.col("model") == "lightgbm") & ~pl.col("is_variant")).sort("step")
        if a.height >= 2:
            points.append(f"**The cart is the strongest signal.** Adding cart behaviour lifts LightGBM from "
                          f"{a['pr_auc_mean'][0]:.3f} to {a['pr_auc_mean'][1]:.3f} PR-AUC, far more than any other "
                          "feature group.")
    if rq4 is not None:
        p = ui.per_k(rq4)
        points.append(f"**It is fast enough for a live shop.** All {p.height} models score a visit within the "
                      f"10 ms budget; the slowest needs {float(p['latency_b1_p95_ms'].max()):.2f} ms.")
    if ex and ex.get("buyers_missed_by_all"):
        m = ex["buyers_missed_by_all"]
        points.append(f"**There is a ceiling.** {m['n'] / m['of']:.0%} of buyers show no buying sign in their first "
                      "clicks, and no model catches them.")
    for p_ in points:
        st.markdown(f"- {p_}")
with right:
    st.subheader("What a shop does with it", anchor=False)
    st.markdown(
        "- **High intent:** keep the path clear (express checkout). No discount needed.\n"
        "- **Warm:** a nudge, such as free shipping or a low-stock message.\n"
        "- **Just browsing:** don't spend coupons; show inspiration instead.\n\n"
        "Try it on the **Live demo** page."
    )
