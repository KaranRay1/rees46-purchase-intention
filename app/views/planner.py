"""Campaign planner: who should get the discount, and what the model is worth in money."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui
from src.serving.campaign import Offer, best_share, gain_table, suppression_value
from src.serving.results import test_predictions

s, K = ui.store(), ui.k()
ink, pal = ui.INK[ui.theme()], ui.SERIES[ui.theme()]
ui.page_header("Campaign planner",
               "Plan a discount campaign with the engine: offer it to everyone except the visitors who are about to "
               "buy anyway, and see what that is worth.", rq="Engine")

rq1 = s.csv("rq1_model_comparison")
if rq1 is None:
    ui.missing("model results", "03_run_experiments.py")
    st.stop()
models = [m for m in ui.models_in(ui.per_k(rq1)) if m != "rule_based"]
ship, _ = team_page.recommended(rq1, s.csv("rq1_pairwise_tests"), s.csv("rq4_efficiency"), K)


@st.cache_data(show_spinner="Loading test-period scores…", max_entries=8)
def scores(mode: str, k: int, model: str) -> pl.DataFrame | None:
    return test_predictions(ui._config(mode), k, model)


# --------------------------------------------------------------------------- inputs
summary = s.json("data_summary") or {}
daily = ui.daily()
default_aov = float(daily["revenue"].sum() / max(daily["buying_sessions"].sum(), 1)) if daily is not None else 300.0
sp = (summary.get("splits") or {}).get(str(K), {})
days = 1.0
if sp:
    a = datetime.fromisoformat(sp["test"]["first_start"])
    b = datetime.fromisoformat(sp["test"]["last_start"])
    days = max((b - a).total_seconds() / 86400, 1.0)
clean = summary.get("cleaning") or {}
sample = clean["sessions_out"] / clean["sessions_in"] if clean.get("sessions_in") else 1.0

left, right = st.columns([1, 2.3], gap="large")
with left:
    st.markdown("<div class='step'>Campaign settings</div>", unsafe_allow_html=True)
    model = st.selectbox("Scoring model", models, index=models.index(ship) if ship in models else 0,
                         format_func=lambda m: ("★ " if m == ship else "") + ui.label(m), key="plan_model",
                         help="★ = the engine's recommended model.")
    discount = st.slider("Discount", 1, 40, 10, format="%d%%", key="plan_discount") / 100
    uplift = st.slider("Extra buyers the offer creates", 0.5, 10.0, 2.0, 0.5, format="%.1f%%", key="plan_uplift",
                       help="Of the visitors who would NOT have bought, the share who buy because of the offer. "
                            "This is an assumption: measure it with an A/B test before relying on it.") / 100
    aov = st.number_input("Average order value ($)", 1.0, 10_000.0, round(default_aov, 0), 10.0, key="plan_aov",
                          help="Default: revenue per buying visit in this dataset.")
    contact = st.number_input("Cost per offer shown ($)", 0.0, 5.0, 0.0, 0.01, format="%.2f", key="plan_contact",
                              help="Message, email or ad cost per visitor who sees the offer.")
    monthly = st.toggle("Whole shop, per month", value=True, key="plan_scale",
                        help=f"The test period covers {days:.0f} days of a {sample:.0%} sample of visits.")

pred = scores(s.mode, K, model)
if pred is None or not pred.height:
    right.info("No saved test predictions for this model and K on this machine.", icon=":material/hourglass_empty:")
    st.stop()

y, sc = pred["y_true"].to_numpy(), pred["score"].to_numpy()
offer = Offer(aov=aov, discount=discount, uplift=uplift, contact_cost=contact)
v = suppression_value(gain_table(y, sc, np.round(np.arange(0, 0.701, 0.01), 2)), offer)
scale = (30 / days) / sample if monthly else 1.0
best = best_share(v)
blanket = float(v["net_blanket"][0])
unit = "per month, whole shop" if monthly else f"{days:.0f}-day test sample"

# --------------------------------------------------------------------------- answer
with right:
    st.markdown("<div class='step'>Recommendation</div>", unsafe_allow_html=True)
    if best["share"] == 0:
        st.markdown("#### Offer it to everyone\nWith these settings, skipping any visitor costs more in lost extra "
                    "sales than it saves in discounts.")
    else:
        st.markdown(f"#### Skip the top {best['share']:.0%} of visits by score, offer to the rest\n"
                    f"In practice: no discount when the engine's score is at least **{best['score_cutoff']:.0%}**.")
    c = st.columns(4)
    c[0].metric("Net value", ui.fmt_money(best["net"] * scale), "per month" if monthly else "test sample",
                delta_color="off", delta_arrow="off")
    gain_vs = (best["net"] - blanket) * scale
    c[1].metric("vs offer to everyone", ("+" if gain_vs >= 0 else "−") + ui.fmt_money(abs(gain_vs)),
                f"{(best['net'] / blanket - 1):+.0%}" if blanket > 0 else "everyone: loss-making",
                delta_color="off", delta_arrow="off")
    c[2].metric("Buy anyway", f"{best['precision']:.0%}" if best["skipped"] else "–",
                f"{ui.fmt_int(best['skipped_buyers'] * scale)} saved", delta_color="off", delta_arrow="off",
                help="Among the skipped visitors, the share who went on to buy without an offer (test period).")
    c[3].metric("Visitors offered", f"{best['offered'] / max(len(y), 1):.0%}",
                f"+{ui.fmt_int(best['extra_orders'] * scale)} orders", delta_color="off", delta_arrow="off")

    fig = go.Figure()
    x = v["share"] * 100
    fig.add_trace(go.Scatter(x=x, y=v["net"] * scale, name=f"Skip the top by {ui.label(model)} score",
                             line=dict(color=pal[0], width=2.5),
                             hovertemplate="skip top %{x:.0f}%: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=v["net_random"] * scale, name="Skip the same number at random",
                             line=dict(color=ink["muted"], width=2, dash="dot"),
                             hovertemplate="skip %{x:.0f}% at random: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[best["share"] * 100], y=[best["net"] * scale], mode="markers", name="Best",
                             marker=dict(size=11, color=pal[0], line=dict(color=ink["surface"], width=2)),
                             hovertemplate="best: skip top %{x:.0f}%<br>%{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=ink["grid"], width=1))
    ui.style(fig, height=330, top=50)
    fig.update_xaxes(title="Visitors skipped (top % by score)", ticksuffix="%", range=[0, 70])
    fig.update_yaxes(title=f"Net value, {unit}", tickprefix="$", tickformat="~s")
    ui.show(fig, key="plan_curve")
    st.caption("At 0 % nobody is skipped (offer to everyone). The gap between the blue and grey lines is what the "
               "model's ranking adds over skipping visitors blindly.")

# --------------------------------------------------------------------------- options table
st.subheader("Compare options", anchor=False)
picks = sorted({0.0, 0.05, 0.1, 0.2, 0.3, float(best["share"])})
tbl = (v.filter(pl.col("share").is_in(picks))
       .select(
           pl.when(pl.col("share") == 0).then(pl.lit("Offer to everyone"))
           .otherwise(pl.format("Skip top {}%", (pl.col("share") * 100).round(0).cast(pl.Int32)))
           .alias("Option"),
           (pl.col("offered") * scale).alias("Offers"),
           (pl.col("skipped_buyers") * scale).alias("Discounts saved"),
           (pl.col("extra_orders") * scale).alias("Extra orders"),
           (pl.col("extra_revenue") * scale).alias("Extra revenue"),
           (pl.col("cost") * scale).alias("Cost"),
           (pl.col("net") * scale).alias("Net value"),
           (pl.col("share") == best["share"]).alias("Best"),
       ))
st.dataframe(tbl, hide_index=True, width="stretch", column_config={
    "Offers": st.column_config.NumberColumn(format="compact"),
    "Discounts saved": st.column_config.NumberColumn(format="compact"),
    "Extra orders": st.column_config.NumberColumn(format="compact"),
    "Extra revenue": st.column_config.NumberColumn(format="dollar"),
    "Cost": st.column_config.NumberColumn(format="dollar", help="Discounts redeemed plus offer costs."),
    "Net value": st.column_config.NumberColumn(format="dollar"),
    "Best": st.column_config.CheckboxColumn(),
})

with st.expander("How this is calculated, and what it assumes", icon=":material/info:"):
    st.markdown(
        f"- **Measured:** who bought in the test period ({ui.fmt_int(len(y))} visits scored after {K} clicks, "
        f"{y.mean():.1%} bought) and how the {ui.label(model)} model ranked them.\n"
        "- **Assumed (your settings):** the discount, the order value, the cost per offer, and the share of "
        "non-buyers the offer converts. The offer is assumed to work equally on every non-buyer.\n"
        "- **Why not target the top scorers?** Under that assumption, the visitors most likely to buy are the "
        "worst to discount: most of them buy anyway, so their discount is pure cost. The engine's value is "
        "knowing whom to *leave out*.\n"
        "- **Before launch:** run an A/B test with a hold-out group to measure the real extra conversion.\n"
        f"- Monthly projection: × {30 / days:.2f} (test days to a month) × {1 / sample:.0f} (sample to whole shop)."
    )
