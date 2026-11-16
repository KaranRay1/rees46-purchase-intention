"""Sales & traffic: the shop's numbers for all months together or any single month, plus the model test window."""

from datetime import datetime

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from src.serving.results import monthly_profile

s = ui.store()
summary = s.json("data_summary")

ui.page_header("Sales & traffic",
               "Visits, orders and revenue from the REES46 shop log. Pick all months or a single month.",
               rq="Business")
if summary is None:
    ui.missing("data summary", "01_prepare_data.py")
    st.stop()

ink = ui.INK[ui.theme()]
pal = ui.SERIES[ui.theme()]
clean = summary.get("cleaning") or {}


@st.cache_data(show_spinner="Counting events per month…")
def months(mode: str) -> pl.DataFrame | None:
    return monthly_profile(ui._config(mode))


mp, daily, pm = months(s.mode), ui.daily(), ui.product_month()
if mp is None or daily is None or not mp.height:
    st.info("The cleaned events for this dataset are not on this machine, so per-month figures are unavailable. "
            f"Run `python scripts/01_prepare_data.py --mode {s.mode}` to create them.",
            icon=":material/hourglass_empty:")
    st.stop()

# --------------------------------------------------------------------------- per-month table (all KPIs)
per_month = (
    daily.group_by("month").agg(pl.col("purchases", "revenue", "views", "carts").sum())
    .join(mp.select("month", "sessions", "buying_sessions", "conversion"), on="month")
    .join(pm.group_by("month").agg(products=pl.col("product_id").n_unique()) if pm is not None
          else mp.select("month", products=pl.lit(None, pl.Int64)), on="month", how="left")
    .with_columns(aov=pl.col("revenue") / pl.col("purchases"))
    .sort("month")
)
month_list = per_month["month"].to_list()
ALL = "All months"

# --------------------------------------------------------------------------- selector
options = [ALL] + month_list
if st.session_state.get("sales_month") not in options:
    st.session_state["sales_month"] = ALL
st.segmented_control("Period", options, key="sales_month", required=True,
                     format_func=lambda m: m if m == ALL else ui.month_name(m, short=True),
                     label_visibility="collapsed")
sel = st.session_state["sales_month"]
note = ui.sample_note()


def totals(df: pl.DataFrame) -> dict:
    return {
        "sessions": int(df["sessions"].sum()), "buying": int(df["buying_sessions"].sum()),
        "orders": int(df["purchases"].sum()), "revenue": float(df["revenue"].sum()),
        "views": int(df["views"].sum()), "carts": int(df["carts"].sum()),
    }


if sel == ALL:
    cur, prev = totals(per_month), None
    products_n = pm["product_id"].n_unique() if pm is not None else None
    period = f"{ui.month_name(month_list[0])} – {ui.month_name(month_list[-1])}" if len(month_list) > 1 \
        else ui.month_name(month_list[0])
else:
    i = month_list.index(sel)
    cur = totals(per_month.filter(pl.col("month") == sel))
    prev = totals(per_month.filter(pl.col("month") == month_list[i - 1])) if i > 0 else None
    products_n = per_month.filter(pl.col("month") == sel)["products"][0]
    period = ui.month_name(sel)
conv = cur["buying"] / max(cur["sessions"], 1)
aov = cur["revenue"] / max(cur["orders"], 1)


prev_name = ui.month_name(month_list[month_list.index(sel) - 1])[:3] if prev else ""


def change(now: float, before: float | None, pct: bool = True) -> str | None:
    if before in (None, 0):
        return None
    return f"{(now / before - 1):+.1%} vs {prev_name}" if pct else f"{(now - before) * 100:+.2f} pp vs {prev_name}"


c = st.columns(6)
c[0].metric("Visits", ui.fmt_int(cur["sessions"]), change(cur["sessions"], prev and prev["sessions"]))
c[1].metric("Conversion", f"{conv:.2%}",
            change(conv, prev and prev["buying"] / max(prev["sessions"], 1), pct=False),
            help="Share of visits that include a purchase.")
c[2].metric("Orders", ui.fmt_int(cur["orders"]), change(cur["orders"], prev and prev["orders"]),
            help="Purchase events (one per product bought).")
c[3].metric("Revenue", ui.fmt_money(cur["revenue"]), change(cur["revenue"], prev and prev["revenue"]),
            help="Sum of the prices of purchased products.")
c[4].metric("Avg. order value", f"${aov:,.0f}",
            change(aov, prev and prev["revenue"] / max(prev["orders"], 1)))
c[5].metric("Products viewed", ui.fmt_int(products_n) if products_n else "–")
st.caption(f"{period}." + (f" {note}" if note else "") + (" Change is shown against the previous month."
                                                           if prev else ""))

# --------------------------------------------------------------------------- trend + funnel
left, right = st.columns([1.7, 1], gap="large")
METRICS = {"Visits": ("sessions", ",.0f"), "Revenue": ("revenue", "$,.0f"), "Orders": ("purchases", ",.0f"),
           "Conversion": ("conversion", ".2%")}


def fv(v: float, fmt: str) -> str:
    """Python formatting for a d3-style format (a leading ``$`` becomes a compact dollar amount)."""
    return ui.fmt_money(v) if fmt.startswith("$") else (ui.fmt_int(v) if fmt == ",.0f" else format(v, fmt))


with left:
    h, t = st.columns([1, 1.5])
    h.subheader("Daily trend" if sel != ALL else "Month by month", anchor=False)
    metric = t.segmented_control("Metric", list(METRICS), default="Visits", key="trend_metric",
                                 label_visibility="collapsed") or "Visits"
    col, fmt = METRICS[metric]
    if sel == ALL:
        d = per_month.with_columns(x=pl.col("month").map_elements(lambda m: ui.month_name(m, True),
                                                                   return_dtype=pl.String))
        fig = go.Figure(go.Bar(
            x=d["x"], y=d[col], marker=dict(color=pal[0], cornerradius=4),
            text=[fv(v, fmt) for v in d[col]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>",
        ))
        ui.style(fig, height=330, legend=False)
        fig.update_yaxes(range=[0, float(d[col].max()) * 1.18])
    else:
        d = daily.filter(pl.col("month") == sel).with_columns(conversion=pl.col("buying_sessions") / pl.col("sessions"))
        fig = go.Figure(go.Scatter(
            x=d["date"], y=d[col], mode="lines", line=dict(color=pal[0], width=2), fill="tozeroy",
            fillcolor="rgba(42,120,214,.10)", hovertemplate=f"%{{x|%a %d %b}}: %{{y:{fmt}}}<extra></extra>",
        ))
        peak = d.sort(col, descending=True).row(0, named=True)
        fig.add_trace(go.Scatter(x=[peak["date"]], y=[peak[col]], mode="markers+text",
                                 marker=dict(size=9, color=pal[0], line=dict(color=ink["surface"], width=2)),
                                 text=[f"peak {fv(peak[col], fmt)}"], textposition="top left",
                                 textfont=dict(color=ink["text"], size=11), hoverinfo="skip"))
        ui.style(fig, height=330, legend=False, hovermode="x unified")
        fig.update_yaxes(rangemode="tozero", range=[0, float(d[col].max()) * 1.2])
    fig.update_yaxes(tickformat=".1%" if col == "conversion" else None, title=None)
    fig.update_xaxes(showgrid=False)
    ui.show(fig, key="trend")

with right:
    st.subheader("Shopping funnel", anchor=False)
    steps = [("Product views", cur["views"]), ("Add to cart", cur["carts"]), ("Purchases", cur["orders"])]
    fig = go.Figure(go.Funnel(
        y=[n for n, _ in steps], x=[v for _, v in steps], textinfo="value+percent initial",
        texttemplate="%{value:,.0f}<br>%{percentInitial:.1%} of views",
        marker=dict(color=[pal[0], pal[3], pal[2]]), connector=dict(fillcolor=ink["neutral"]),
        textfont=dict(size=12), hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    ui.style(fig, height=330, legend=False, left=110, top=10)
    ui.show(fig, key="funnel")

# --------------------------------------------------------------------------- departments
if pm is not None:
    st.subheader(f"Revenue by department · {period}", anchor=False)
    dsel = pm if sel == ALL else pm.filter(pl.col("month") == sel)
    dep = (dsel.group_by("department").agg(pl.col("revenue", "views", "purchases").sum(),
                                           products=pl.col("product_id").n_unique())
           .with_columns(conversion=pl.col("purchases") / pl.col("views"),
                         share=pl.col("revenue") / pl.col("revenue").sum())
           .sort("revenue", descending=True))
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        top = dep.head(10).sort("revenue")
        fig = go.Figure(go.Bar(
            y=[x.replace("_", " ").capitalize() for x in top["department"]], x=top["revenue"], orientation="h",
            marker=dict(color=[ink["grid"] if x == "unknown" else pal[0] for x in top["department"]], cornerradius=4),
            text=[f"{ui.fmt_money(v)} · {sh:.0%}" for v, sh in zip(top["revenue"], top["share"])],
            textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=11),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
        ))
        ui.style(fig, height=380, legend=False, left=110, top=10)
        fig.update_xaxes(range=[0, float(top["revenue"].max()) * 1.3], title="Revenue")
        fig.update_yaxes(showgrid=False)
        ui.show(fig, key="dept")
        st.caption("“Unknown” (grey): products the shop never gave a category.")
    with right:
        st.dataframe(
            dep.select(pl.col("department").str.replace("_", " ").str.to_titlecase(), "products", "views",
                       "purchases", "conversion", "revenue"),
            hide_index=True, width="stretch", height=380,
            column_config={
                "department": "Department", "products": st.column_config.NumberColumn("Products", format="localized"),
                "views": st.column_config.NumberColumn("Views", format="compact"),
                "purchases": st.column_config.NumberColumn("Orders", format="localized"),
                "conversion": st.column_config.NumberColumn("View → buy", format="percent"),
                "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
            })

# --------------------------------------------------------------------------- month table
st.subheader("Month-by-month summary", anchor=False)
total = per_month.select(
    pl.lit("Total").alias("month"), pl.col("sessions", "buying_sessions", "purchases", "revenue").sum(),
    (pl.col("buying_sessions").sum() / pl.col("sessions").sum()).alias("conversion"),
    (pl.col("revenue").sum() / pl.col("purchases").sum()).alias("aov"),
    pl.lit(pm["product_id"].n_unique() if pm is not None else None, pl.Int64).alias("products"),
)
table = pl.concat([
    per_month.select("month", "sessions", "buying_sessions", "purchases", "revenue", "conversion", "aov", "products")
    .with_columns(pl.col("month").map_elements(ui.month_name, return_dtype=pl.String)),
    total.select("month", "sessions", "buying_sessions", "purchases", "revenue", "conversion", "aov", "products")
    .with_columns(pl.col("products").cast(per_month["products"].dtype)),
], how="vertical_relaxed")
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "month": "Month", "sessions": st.column_config.NumberColumn("Visits", format="localized"),
    "buying_sessions": st.column_config.NumberColumn("Buying visits", format="localized"),
    "purchases": st.column_config.NumberColumn("Orders", format="localized"),
    "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
    "conversion": st.column_config.NumberColumn("Conversion", format="percent"),
    "aov": st.column_config.NumberColumn("Avg. order", format="dollar"),
    "products": st.column_config.NumberColumn("Products viewed", format="localized"),
})
st.download_button("Download as CSV", table.write_csv(), file_name=f"sales_by_month_{s.mode}.csv",
                   mime="text/csv", icon=":material/download:")

# --------------------------------------------------------------------------- model window
st.divider()
st.subheader("How the models were trained and tested on this data", anchor=False)
left, right = st.columns([1.4, 1], gap="large")
with left:
    sp = summary.get("splits", {}).get(str(ui.k()))
    if sp:
        fig = go.Figure()
        names = {"train": "Train (70%)", "val": "Tune (15%)", "test": "Test (15%)"}
        for i, part in enumerate(("train", "val", "test")):
            p = sp[part]
            a = datetime.fromisoformat(p["first_start"].replace("Z", "+00:00"))
            b = datetime.fromisoformat(p["last_start"].replace("Z", "+00:00"))
            fig.add_trace(go.Bar(
                y=["split"], x=[(b - a).total_seconds() * 1000], base=[a], orientation="h", name=names[part],
                marker=dict(color=[pal[0], pal[3], pal[1]][i], line=dict(color=ink["surface"], width=2)),
                text=[part.replace("val", "tune").capitalize()], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#ffffff", size=12),
                hovertemplate=f"{names[part]}<br>{a:%d %b %Y} – {b:%d %b %Y}<br>{p['sessions']:,} visits · "
                              f"{p['positive_rate']:.1%} buy<extra></extra>",
            ))
        ui.style(fig, height=150, legend=False, left=16, top=10, bottom=40, barmode="overlay")
        fig.update_yaxes(visible=False)
        fig.update_xaxes(type="date", showgrid=False)
        ui.show(fig, key="split")
        ui.legend_items((f"{names[x]}: {sp[x]['first_start'][:10]} → {sp[x]['last_start'][:10]}, "
                         f"{sp[x]['sessions']:,} visits", c_) for x, c_ in zip(("train", "val", "test"),
                                                                              (pal[0], pal[3], pal[1])))
    st.caption("The models learn from earlier months and are judged on later months they never saw, which is "
               "how the engine runs in production.")
with right:
    st.markdown(
        "Each visit is scored after its first **K** clicks (3, 5 or 10). Visits that already bought within those "
        "clicks are left out, so the answer is never in the input. From the clicks we compute **18 features** "
        "(activity, cart behaviour, cart/view ratio, categories, price & time); the GRU reads the raw clicks."
    )
    sk = sorted(summary.get("sessionize", {}).items(), key=lambda kv: int(kv[0]))
    st.dataframe([{"K": int(k_), "Visits usable": v["eligible"], "Buy later": v["positive_rate"]} for k_, v in sk],
                 hide_index=True, width="stretch",
                 column_config={"Visits usable": st.column_config.NumberColumn(format="localized"),
                                "Buy later": st.column_config.NumberColumn(format="percent")})
