"""Monitoring: does the engine keep working week after week, and are its inputs drifting?"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui

s, K = ui.store(), ui.k()
ink, pal = ui.INK[ui.theme()], ui.SERIES[ui.theme()]
ui.page_header("Monitoring",
               "Accuracy week by week on the weeks after training, and drift of the engine's inputs month by month.",
               rq="Engine")

weekly = s.csv("monitoring_weekly")
drift = s.csv("monitoring_drift")
rq1 = s.csv("rq1_model_comparison")
if weekly is None or drift is None or rq1 is None:
    ui.missing("monitoring tables", "07_monitor.py")
    st.stop()

weekly, drift = ui.per_k(weekly), ui.per_k(drift)
if "period" not in drift.columns:                       # tables written before weekly drift existed
    drift = drift.rename({"month": "period"}).with_columns(granularity=pl.lit("month"))
by_week = drift["granularity"][0] == "week" if drift.height else False


def period_name(p: str, short: bool = False) -> str:
    if by_week:
        from datetime import date

        d = date.fromisoformat(p)
        return f"wk {d:%d %b}" if short else f"the week of {d:%d %b %Y}"
    return ui.month_name(p, short)


if not weekly.height:
    st.info(f"No monitoring results for K={K}. Run `python scripts/07_monitor.py --ks {K}`.")
    st.stop()
ship, _ = team_page.recommended(rq1, s.csv("rq1_pairwise_tests"), s.csv("rq4_efficiency"), K)
models = ui.models_in(weekly)

# --------------------------------------------------------------------------- controls + KPIs
c1, c2 = st.columns([1.2, 3])
if st.session_state.get("mon_model") not in models:     # a choice left over from another dataset or K
    st.session_state.pop("mon_model", None)
model = c1.selectbox("Model", models, index=models.index(ship) if ship in models else 0, key="mon_model",
                     format_func=lambda m: ("★ " if m == ship else "") + ui.label(m))
w = weekly.filter(pl.col("model") == model).with_columns(pl.col("week").str.to_date()).sort("week")
if w.drop_nulls("pr_auc").is_empty():
    st.info(f"No weekly results for {ui.label(model)} at K={K}. Pick another model, or rerun "
            f"`python scripts/07_monitor.py --ks {K}`.")
    st.stop()
overall = ui.per_k(rq1).filter(pl.col("model") == model).drop_nulls("pr_auc_mean")
ref = float(overall["pr_auc_mean"][0]) if overall.height else float(w["pr_auc"].mean())
full_weeks = w.filter(pl.col("sessions") >= w["sessions"].max() * 0.3)       # ignore part-weeks at the edges
low = full_weeks.sort("pr_auc").row(0, named=True)
alert_weeks = full_weeks.filter(pl.col("pr_auc_high") < ref)
after = drift.filter(pl.col("split") != "train")
n_major = after.filter(pl.col("status") == "major")["feature"].n_unique()
n_mod = after.filter(pl.col("status") == "moderate")["feature"].n_unique()

k = st.columns(4)
k[0].metric("Weeks monitored", f"{w.height}", f"{w['week'].min():%d %b} – {w['week'].max():%d %b %Y}",
            delta_color="off", delta_arrow="off")
k[1].metric("Weekly PR-AUC range", f"{full_weeks['pr_auc'].min():.2f} – {full_weeks['pr_auc'].max():.2f}",
            f"test-period average {ref:.3f}", delta_color="off", delta_arrow="off")
k[2].metric("Weeks clearly below average", f"{alert_weeks.height}",
            f"lowest: week of {low['week']:%d %b} ({low['pr_auc']:.2f})" if alert_weeks.height else "none",
            delta_color="off", delta_arrow="off",
            help="Weeks whose whole 95 % confidence interval lies below the test-period PR-AUC.")
k[3].metric("Inputs drifting", "None" if not n_major + n_mod else f"{n_major + n_mod} features",
            f"{n_major} major · {n_mod} moderate, after training", delta_color="off", delta_arrow="off",
            help="Features whose PSI against the training period exceeds 0.25 (major) or 0.10 (moderate) in any "
                 "month (or week, for one-month data) after training.")

# --------------------------------------------------------------------------- weekly accuracy
left, right = st.columns([1.6, 1], gap="large")
with left:
    st.subheader("Accuracy week by week", anchor=False)
    fig = go.Figure()
    x = w["week"].to_list()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=w["pr_auc_high"].to_list() + w["pr_auc_low"].to_list()[::-1],
                             fill="toself", fillcolor="rgba(42,120,214,.15)", line=dict(width=0), hoverinfo="skip",
                             name="95% CI"))
    fig.add_trace(go.Scatter(x=x, y=w["pr_auc"], mode="lines+markers", name=f"{ui.label(model)} PR-AUC",
                             line=dict(color=pal[0], width=2), marker=dict(size=7),
                             customdata=list(zip(w["pr_auc_low"], w["pr_auc_high"], w["sessions"], w["split"],
                                                 strict=True)),
                             hovertemplate="week of %{x|%d %b}: %{y:.3f} (95% CI %{customdata[0]:.3f}–"
                                           "%{customdata[1]:.3f})<br>%{customdata[2]:,} visits · %{customdata[3]}"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=w["base_rate"], mode="lines", name="Random guessing (buyer rate)",
                             line=dict(color=ink["muted"], width=1.5, dash="dot"),
                             hovertemplate="buyer rate %{y:.1%}<extra></extra>"))
    fig.add_hline(y=ref, line=dict(color=pal[1], width=1.5, dash="dash"))
    test_start = w.filter(pl.col("split") == "test")["week"].min()
    if test_start is not None:
        fig.add_vline(x=test_start, line=dict(color=ink["grid"], width=1))
        fig.add_annotation(x=test_start, y=1, yref="paper", text="test period →", showarrow=False, xanchor="left",
                           font=dict(color=ink["muted"], size=11))
    ui.style(fig, height=360, top=50)
    fig.update_yaxes(title="PR-AUC", range=[0, max(float(w["pr_auc_high"].max()) * 1.15, 0.1)])
    fig.update_xaxes(showgrid=False)
    ui.show(fig, key="mon_weekly")
    st.caption(f"Dashed line: the model's test-period PR-AUC ({ref:.3f}). Validation and test weeks both come after "
               "the training period, so every point is out-of-sample.")
with right:
    st.subheader("What moved it", anchor=False)
    corr = full_weeks.select(pl.corr("pr_auc", "base_rate")).item() if full_weeks.height > 2 else None
    st.markdown(
        f"- **Lowest week:** {low['week']:%d %b %Y}, PR-AUC {low['pr_auc']:.3f} with a buyer rate of "
        f"{low['base_rate']:.1%}.\n"
        + (f"- Weekly accuracy follows the share of buyers (correlation **{corr:.2f}**): fewer buyers means a harder "
           "ranking problem, not necessarily a broken model. Lift over random stays between "
           f"**{full_weeks['lift'].min():.1f}×** and **{full_weeks['lift'].max():.1f}×**.\n" if corr is not None else "")
        + f"- **Scores are rankings, not probabilities.** The average score is {w['mean_score'].mean():.2f} while "
        f"{w['buyers'].sum() / w['sessions'].sum():.1%} of visits buy: the models were trained with class balancing. "
        "Use the score to rank and to compare with the tuned threshold, not as a literal chance."
    )
    st.markdown("<div class='card'><div class='step'>Alert rules</div><p>Investigate when the weekly 95 % CI "
                "falls below the test-period PR-AUC for <b>two weeks in a row</b>, or when any input's PSI "
                "exceeds <b>0.25</b>. Retrain monthly on a rolling window.</p></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- drift
st.divider()
st.subheader(f"Input drift by {'week' if by_week else 'month'} (PSI against the training period)", anchor=False)
months = sorted(drift["period"].unique().to_list())
feats = (drift.group_by("feature").agg(pl.col("psi").max()).sort("psi", descending=True)["feature"].to_list())
grid = {(r["feature"], r["period"]): r["psi"] for r in drift.iter_rows(named=True)}
z = [[grid.get((f, m)) for m in months] for f in feats]
split_of = dict(drift.select("period", "split").unique().iter_rows())
labels = [f"{period_name(m, True)}<br><span style='font-size:10px'>{split_of.get(m, '')}</span>" for m in months]
fig = go.Figure(go.Heatmap(
    z=z, x=labels, y=feats, zmin=0, zmax=0.3,
    colorscale=[[0, ink["surface"]], [0.33, "#9ec5f4"], [0.83, "#256abf"], [1, "#0d366b"]],
    text=[[f"{v:.2f}" if v is not None and v == v else "n/a" for v in row] for row in z], texttemplate="%{text}",
    textfont=dict(size=10), xgap=2, ygap=2,
    colorbar=dict(title="PSI", tickvals=[0, 0.1, 0.25], ticktext=["0", "0.10 moderate", "0.25 major"], len=0.8),
    hovertemplate="%{y}, %{x}: PSI %{z:.3f}<extra></extra>",
))
ui.style(fig, height=80 + 24 * len(feats), legend=False, left=190, top=10)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=False, side="top")
ui.show(fig, key="mon_drift")
after_known = after.filter(pl.col("psi").is_not_null())
worst = after_known.sort("psi", descending=True).row(0, named=True) if after_known.height else None
st.caption("PSI below 0.10 is stable, 0.10–0.25 a moderate shift, above 0.25 a major shift. "
           + ("“n/a”: day of week is not compared for part-weeks at the start or end of the data. " if by_week else "")
           + (f"Largest shift after training: {worst['feature']} in {period_name(worst['period'])} "
              f"(PSI {worst['psi']:.3f}, {worst['status']})." if worst else ""))
