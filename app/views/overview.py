"""Overview: the shop at a glance, the scoring engine in production, and what the models tell the business."""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui
from src.serving.catalog import products as collapse
from src.serving.results import monthly_profile

s, K = ui.store(), ui.k()
summary = s.json("data_summary")
rq1 = s.csv("rq1_model_comparison")
rq4 = s.csv("rq4_efficiency")
pairs = s.csv("rq1_pairwise_tests")
abl = s.csv("rq3_ablation")
ex = s.json("rq5_explain_summary")
ink, pal = ui.INK[ui.theme()], ui.SERIES[ui.theme()]

ui.page_header("Overview",
               "The shop at a glance and the purchase-intent engine that scores every visit after its first few "
               "clicks.", rq="Business")


@st.cache_data(show_spinner=False)
def months(mode: str) -> pl.DataFrame | None:
    return monthly_profile(ui._config(mode))


mp, daily, pm = months(s.mode), ui.daily(), ui.product_month()

# --------------------------------------------------------------------------- KPIs
clean = (summary or {}).get("cleaning") or {}
c = st.columns(5)
if clean:
    c[0].metric("Visits", ui.fmt_int(clean["sessions_out"]), f"{ui.fmt_int(clean['rows_out'])} clicks",
                delta_color="off", delta_arrow="off")
    c[1].metric("Conversion", f"{clean['purchase_sessions_out'] / max(clean['sessions_out'], 1):.2%}",
                "visits that buy", delta_color="off", delta_arrow="off")
if daily is not None:
    c[2].metric("Revenue", ui.fmt_money(daily["revenue"].sum()), f"{ui.fmt_int(daily['purchases'].sum())} orders",
                delta_color="off", delta_arrow="off")
if pm is not None:
    c[3].metric("Products", ui.fmt_int(pm["product_id"].n_unique()),
                f"{pm['brand'].n_unique():,} brands", delta_color="off", delta_arrow="off")
if mp is not None and mp.height:
    c[4].metric("Period", f"{mp.height} month" + ("s" if mp.height > 1 else ""),
                ui.month_name(mp["month"][0]) if mp.height == 1
                else f"{ui.month_name(mp['month'][0], True)} – {ui.month_name(mp['month'][-1], True)}",
                delta_color="off", delta_arrow="off")
if ui.sample_note():
    st.caption(ui.sample_note())

# --------------------------------------------------------------------------- trend + engine
left, right = st.columns([1.6, 1], gap="large")
with left:
    if daily is not None and mp is not None:
        h, t = st.columns([1.2, 1])
        h.subheader("Monthly performance", anchor=False)
        metric = t.segmented_control("Metric", ["Revenue", "Visits", "Conversion"], default="Revenue",
                                     key="ov_metric", label_visibility="collapsed") or "Revenue"
        per = (daily.group_by("month").agg(pl.col("revenue").sum()).join(mp, on="month").sort("month"))
        col = {"Revenue": "revenue", "Visits": "sessions", "Conversion": "conversion"}[metric]
        txt = {"revenue": ui.fmt_money, "sessions": ui.fmt_int, "conversion": lambda v: f"{v:.1%}"}[col]
        fig = go.Figure(go.Bar(
            x=[ui.month_name(m, True) for m in per["month"]], y=per[col], marker=dict(color=pal[0], cornerradius=4),
            text=[txt(v) for v in per[col]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11), hovertemplate="%{x}: %{text}<extra></extra>",
        ))
        ui.style(fig, height=300, legend=False, top=10)
        fig.update_yaxes(range=[0, float(per[col].max()) * 1.2], tickformat=".0%" if col == "conversion" else None)
        fig.update_xaxes(showgrid=False)
        ui.show(fig, key="ov_trend")
with right:
    st.subheader("Scoring engine", anchor=False)
    if rq1 is None:
        ui.missing("model results", "03_run_experiments.py")
    else:
        ship, tied = team_page.recommended(rq1, pairs, rq4, K)
        r = ui.per_k(rq1).filter(pl.col("model") == ship).row(0, named=True)
        e = ui.per_k(rq4).filter(pl.col("model") == ship) if rq4 is not None else None
        lat = f"{e['latency_b1_p95_ms'][0]:.2f} ms" if e is not None and e.height else "–"
        team = ui.TEAMS[ui.team_of(ship)]
        st.markdown(
            f"<div class='card'><div class='step'>In production · scores after {K} clicks</div>"
            f"<h4 style='font-size:1.3rem;margin-top:.2rem'>{ui.label(ship)}</h4>"
            f"<p>{team['title']} · {team['author']}</p>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin-top:.9rem'>"
            f"<div><div class='step'>PR-AUC</div><div style='font-size:1.25rem;font-weight:700'>{r['pr_auc_mean']:.3f}"
            f"</div></div><div><div class='step'>vs random</div><div style='font-size:1.25rem;font-weight:700'>"
            f"{r['pr_auc_lift_mean']:.1f}×</div></div><div><div class='step'>Latency p95</div>"
            f"<div style='font-size:1.25rem;font-weight:700'>{lat}</div></div></div>"
            f"<p style='margin-top:.9rem;font-size:.83rem'>Chosen as the fastest of the {len(tied)} models that are "
            f"statistically tied for the best accuracy.</p></div>",
            unsafe_allow_html=True)
        st.page_link("views/predict.py", label="Score a visit now", icon=":material/bolt:")

# --------------------------------------------------------------------------- departments + top products
if pm is not None:
    st.write("")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Revenue by department", anchor=False)
        dep = (pm.filter(pl.col("department") != "unknown").group_by("department").agg(pl.col("revenue").sum())
               .sort("revenue", descending=True).head(6).sort("revenue"))
        fig = go.Figure(go.Bar(
            y=[d.replace("_", " ").capitalize() for d in dep["department"]], x=dep["revenue"], orientation="h",
            marker=dict(color=pal[0], cornerradius=4), text=[ui.fmt_money(v) for v in dep["revenue"]],
            textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=11),
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        ))
        ui.style(fig, height=280, legend=False, left=100, top=10)
        fig.update_xaxes(range=[0, float(dep["revenue"].max()) * 1.3], showticklabels=False, showgrid=False)
        fig.update_yaxes(showgrid=False)
        ui.show(fig, key="ov_dept")
    with right:
        st.subheader("Top products", anchor=False)
        top = collapse(pm).sort("revenue", descending=True).head(6)
        st.dataframe(top.select("product_id", "category_code", "brand", "purchases", "revenue"), hide_index=True,
                     width="stretch", column_config={
                         "product_id": st.column_config.NumberColumn("Product ID", format="%d"),
                         "category_code": "Category", "brand": "Brand",
                         "purchases": st.column_config.NumberColumn("Orders", format="localized"),
                         "revenue": st.column_config.NumberColumn("Revenue", format="dollar")})
        st.page_link("views/products.py", label="Open the product catalogue", icon=":material/inventory_2:")

# --------------------------------------------------------------------------- insights + playbook
st.write("")
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("What the models tell us", anchor=False)
    points = []
    if abl is not None:
        a = ui.per_k(abl).filter((pl.col("model") == "lightgbm") & ~pl.col("is_variant")).sort("step")
        if a.height >= 2:
            points.append(f"**The cart is the strongest buying signal.** Adding cart behaviour lifts the model from "
                          f"{a['pr_auc_mean'][0]:.3f} to {a['pr_auc_mean'][1]:.3f} PR-AUC, more than any other "
                          "signal.")
    if rq1 is not None:
        wins = [m for m in ("random_forest", "lightgbm", "mlp", "tabnet")
                if (r_ := team_page.significant_vs(pairs, K, m, "logistic_regression")) and r_[1] and r_[0] > 0]
        if wins:
            points.append(f"**Non-linear models are worth it.** {', '.join(ui.label(m) for m in wins)} beat the "
                          "linear baseline, and the gap is statistically significant.")
        points.append("**Machine learning and deep learning reach similar accuracy**, so the engine is chosen on "
                      "speed and simplicity (see Model comparison).")
    if rq4 is not None:
        p = ui.per_k(rq4)
        points.append(f"**Real-time ready.** All {p.height} models score a visit inside a 10 ms budget; the slowest "
                      f"needs {float(p['latency_b1_p95_ms'].max()):.2f} ms.")
    if ex and ex.get("buyers_missed_by_all"):
        m = ex["buyers_missed_by_all"]
        points.append(f"**Known limit:** {m['n'] / m['of']:.0%} of buyers show no buying sign in their first clicks "
                      "and are missed by every model.")
    for p_ in points:
        st.markdown(f"- {p_}")
with right:
    st.subheader("How the shop uses a score", anchor=False)
    for colour, name, body in ((ui.STATUS["good"], "High intent", "Keep the path clear: express checkout, no "
                                "discount needed."),
                               (ui.STATUS["warning"], "Warm", "Nudge: free shipping, low-stock note, cart reminder."),
                               (ui.STATUS["serious"], "Just browsing", "Save the coupon; show inspiration and "
                                "similar products.")):
        st.markdown(f"<div style='display:flex;gap:.7rem;align-items:flex-start;margin:.15rem 0 .7rem 0'>"
                    f"<span style='width:10px;height:10px;border-radius:50%;background:{colour};margin-top:.4rem;"
                    f"flex:none'></span><div><b>{name}</b><br><span style='opacity:.75'>{body}</span></div></div>",
                    unsafe_allow_html=True)
