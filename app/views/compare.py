"""Machine learning vs deep learning: who wins, by how much, at what cost, and which model to ship."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui

s, K = ui.store(), ui.k()
rq1 = s.csv("rq1_model_comparison")
pairs = s.csv("rq1_pairwise_tests")
eff = s.csv("rq4_efficiency")
summary = s.json("rq5_explain_summary")

ui.page_header("Model comparison",
               "Machine learning vs deep learning on the same visits, test period and metric, and which model "
               "the engine runs.",
               rq="Models")
if rq1 is None:
    ui.missing("model results", "03_run_experiments.py")
    st.stop()

ink = ui.INK[ui.theme()]
tcol = ui.team_colours()
part = ui.per_k(rq1)
learned = part.filter(pl.col("model") != "rule_based")


def best(team: str, df: pl.DataFrame = part) -> dict | None:
    d = df.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]) & (pl.col("model") != "rule_based"))
    return d.sort("pr_auc_mean", descending=True).row(0, named=True) if d.height else None


bm, bd = best("ml"), best("dl")

# --------------------------------------------------------------------------- verdict
if bm and bd:
    test = team_page.significant_vs(pairs, K, bm["model"], bd["model"])
    diff = bm["pr_auc_mean"] - bd["pr_auc_mean"]
    winner = "ml" if diff > 0 else "dl"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Best {ui.TEAMS['ml']['title'].lower()} · {ui.TEAMS['ml']['author']}", ui.label(bm["model"]),
              f"PR-AUC {bm['pr_auc_mean']:.3f}", delta_color="off", delta_arrow="off")
    c2.metric(f"Best {ui.TEAMS['dl']['title'].lower()} · {ui.TEAMS['dl']['author']}", ui.label(bd["model"]),
              f"PR-AUC {bd['pr_auc_mean']:.3f}", delta_color="off", delta_arrow="off")
    if test and test[1]:
        verdict = f"{winner.upper()}, by {abs(diff):.3f}"
        note = "Small, but statistically significant."
    else:
        verdict = "A tie"
        note = "The difference is not statistically significant."
    c3.metric(f"Winner after {K} clicks", verdict, note, delta_color="off", delta_arrow="off",
              help="Paired bootstrap on the same test sessions, Holm-corrected for multiple comparisons.")

# --------------------------------------------------------------------------- all models
left, right = st.columns([1.2, 1], gap="large")
with left:
    st.subheader("All models, one scale", anchor=False)
    d = part.sort("pr_auc_mean")
    fig = go.Figure()
    for team in ("ml", "dl"):
        x = d.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]))
        fig.add_trace(go.Bar(
            y=[ui.label(m) for m in x["model"]], x=x["pr_auc_mean"], orientation="h",
            name=f"{ui.TEAMS[team]['title']} ({ui.TEAMS[team]['author']})",
            marker=dict(color=tcol[team], cornerradius=4),
            text=[f"{v:.3f}" for v in x["pr_auc_mean"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=12),
            hovertemplate="<b>%{y}</b><br>PR-AUC %{x:.4f}<extra></extra>",
        ))
    ui.style(fig, height=380, left=150, top=50, barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=[ui.label(m) for m in d["model"]], showgrid=False)
    fig.update_xaxes(range=[0, float(d["pr_auc_mean"].max()) * 1.2], title=f"PR-AUC after {K} clicks")
    ui.show(fig, key="all")
with right:
    st.subheader("Across K (clicks seen)", anchor=False)
    rows = []
    for k_ in sorted(set(rq1["k"].to_list())):
        sub = rq1.filter(pl.col("k") == k_)
        a, b = best("ml", sub), best("dl", sub)
        if a and b:
            t = team_page.significant_vs(pairs, k_, a["model"], b["model"])
            w = "tie" if not (t and t[1]) else ("ML" if t[0] > 0 else "DL")
            rows.append({"K": k_, "Best ML": f"{ui.label(a['model'])} {a['pr_auc_mean']:.3f}",
                         "Best DL": f"{ui.label(b['model'])} {b['pr_auc_mean']:.3f}", "Winner": w})
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"K": st.column_config.NumberColumn(width=40),
                                "Winner": st.column_config.TextColumn(width=60)})
    if bm and bd:
        gap = abs(bm["pr_auc_mean"] - bd["pr_auc_mean"])
        lift = min(bm["pr_auc_lift_mean"], bd["pr_auc_lift_mean"])
        st.markdown(f"**In short:** at K={K} the two teams' best models are {gap:.3f} PR-AUC apart and both rank "
                    f"buyers at least {lift:.1f}× better than random. The choice between them comes down to speed "
                    "and simplicity.")

# --------------------------------------------------------------------------- speed vs accuracy
st.divider()
left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("Accuracy vs speed", anchor=False)
    if eff is None:
        ui.missing("speed benchmarks", "05_benchmark_latency.py")
    else:
        e = ui.per_k(eff).filter(pl.col("model") != "rule_based")
        fig = go.Figure()
        by_x = e.sort("latency_b1_p95_ms")
        pos, prev = {}, None
        for m, y in zip(by_x["model"], by_x["test_pr_auc_primary"]):   # label above if higher than its left neighbour
            pos[m] = "top right" if prev is None or y >= prev else "bottom right"
            prev = y
        for team in ("ml", "dl"):
            x = e.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]))
            fig.add_trace(go.Scatter(
                x=x["latency_b1_p95_ms"], y=x["test_pr_auc_primary"], mode="markers+text",
                name=ui.TEAMS[team]["title"], text=[ui.label(m) for m in x["model"]],
                textposition=[pos[m] for m in x["model"]],
                textfont=dict(color=ink["text"], size=11),
                marker=dict(size=13, color=tcol[team], line=dict(color=ink["surface"], width=2)),
                hovertemplate="<b>%{text}</b><br>%{x:.3f} ms · PR-AUC %{y:.4f}<extra></extra>",
            ))
        ui.style(fig, height=380, top=50)
        xs = e["latency_b1_p95_ms"]
        fig.update_xaxes(type="log", title="Time to score one session, p95 (ms, log scale)",
                         range=[float(np.log10(max(xs.min(), 1e-4) / 3)), float(np.log10(xs.max() * 12))])
        ys = e["test_pr_auc_primary"]
        fig.update_yaxes(title="PR-AUC", range=[float(ys.min()) - 0.01, float(ys.max()) + 0.012])
        ui.show(fig, key="speed")
        st.caption("Up and to the left is better. Everything is far below the 10 ms real-time budget.")
with right:
    st.subheader("What we would ship", anchor=False)
    if eff is not None and bm and bd:
        e = ui.per_k(eff)
        ship, tied = team_page.recommended(rq1, pairs, eff, K)
        other = [m for m in tied if ui.team_of(m) != ui.team_of(ship)]
        alt = min(other, key=lambda m: float(e.filter(pl.col("model") == m)["latency_b1_p95_ms"][0])) if other \
            else (bd if ui.team_of(ship) == "ml" else bm)["model"]

        def blurb(m: str) -> str:
            r, x = part.filter(pl.col("model") == m), e.filter(pl.col("model") == m)
            why = ("exact per-feature explanations (TreeSHAP)" if m in ("lightgbm", "random_forest")
                   else "explanations through SHAP")
            return (f"**{ui.label(m)}** ({ui.TEAMS[ui.team_of(m)]['title'].lower()}): PR-AUC {r['pr_auc_mean'][0]:.3f}, "
                    f"{x['latency_b1_p95_ms'][0]:.2f} ms per session, {x['model_size_mb'][0]:.1f} MB, {why}.")

        st.markdown(f"Statistically tied for the top at K={K}: {', '.join(ui.label(m) for m in tied)}. Among those "
                    "we pick the fastest.")
        st.markdown(f"**Default engine:** {blurb(ship)}\n\n**Alternative from the other team:** {blurb(alt)}")
        st.caption("Both are available in the live demo.")

# --------------------------------------------------------------------------- shared limit
if summary and summary.get("buyers_missed_by_all"):
    st.divider()
    st.subheader("A limit both teams hit", anchor=False)
    m = summary["buyers_missed_by_all"]
    c1, c2 = st.columns([1, 2.2])
    c1.metric("Buyers no model catches", f"{m['n'] / m['of']:.0%}", f"{m['n']:,} of {m['of']:,} test buyers",
              delta_color="off", delta_arrow="off")
    c2.markdown("These shoppers bought later, but their first clicks look exactly like browsing: none of them "
                "added anything to the cart. No model, linear, tree or neural, can tell them apart yet. "
                "That is a property of the data in the first few clicks, not a weakness of one method.")
