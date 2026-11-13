"""The data: which months, how much, and how the fair (time-ordered) test is set up."""

from datetime import datetime

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from src.serving.results import monthly_profile

s = ui.store()
summary = s.json("data_summary")

ui.page_header("The data",
               "REES46: the public click log of a large multi-category online shop. Every view, add-to-cart and "
               "purchase, with time, product, category, brand and price.", rq="1 · Research")
if summary is None:
    ui.missing("data summary", "01_prepare_data.py")
    st.stop()

ink = ui.INK[ui.theme()]
pal = ui.SERIES[ui.theme()]
clean = summary.get("cleaning") or {}


@st.cache_data(show_spinner="Counting events per month…")
def months(mode: str) -> pl.DataFrame | None:
    return monthly_profile(ui._config(mode))


mp = months(s.mode)
c1, c2, c3, c4 = st.columns(4)
if mp is not None and mp.height:
    first = datetime.strptime(mp["month"][0], "%Y-%m").strftime("%b %Y")
    last = datetime.strptime(mp["month"][-1], "%Y-%m").strftime("%b %Y")
    c1.metric("Months", f"{mp.height}", f"{first} – {last}" if mp.height > 1 else first, delta_color="off", delta_arrow="off")
if clean:
    c2.metric("Clicks (events)", ui.fmt_int(clean["rows_out"]), delta_color="off", delta_arrow="off")
    c3.metric("Shop visits (sessions)", ui.fmt_int(clean["sessions_out"]), delta_color="off", delta_arrow="off")
    c4.metric("Visits with a purchase", f"{clean['purchase_sessions_out'] / max(clean['sessions_out'], 1):.1%}",
              f"{ui.fmt_int(clean['purchase_sessions_out'])} visits", delta_color="off", delta_arrow="off")
    if clean.get("removed", {}).get("session_sampling"):
        st.caption(f"A random {clean['sessions_out'] / max(clean['sessions_in'], 1):.0%} of all visits (whole visits, "
                   "chosen by hash) keeps the multi-month run demo-sized; rates stay representative.")

# --------------------------------------------------------------------------- per month
if mp is not None and mp.height:
    labels = [datetime.strptime(m, "%Y-%m").strftime("%b %y") for m in mp["month"]]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Visits per month", anchor=False)
        fig = go.Figure(go.Bar(
            x=labels, y=mp["sessions"], marker=dict(color=pal[0], cornerradius=4),
            text=[ui.fmt_int(v) for v in mp["sessions"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            hovertemplate="%{x}: %{y:,} visits<extra></extra>",
        ))
        ui.style(fig, height=300, legend=False)
        fig.update_yaxes(title="Visits", range=[0, float(mp["sessions"].max()) * 1.2])
        fig.update_xaxes(showgrid=False)
        ui.show(fig, key="m_sessions")
    with right:
        st.subheader("Share of visits that buy", anchor=False)
        fig = go.Figure(go.Scatter(
            x=labels, y=mp["conversion"], mode="lines+markers+text", line=dict(color=pal[2], width=2),
            marker=dict(size=9, line=dict(color=ink["surface"], width=2)),
            text=[f"{v:.1%}" for v in mp["conversion"]], textposition="top center",
            textfont=dict(color=ink["text"], size=11), hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        ))
        ui.style(fig, height=300, legend=False)
        fig.update_yaxes(title="Conversion", tickformat=".0%", range=[0, float(mp["conversion"].max()) * 1.35])
        fig.update_xaxes(showgrid=False)
        ui.show(fig, key="m_conv")
    if mp.height > 1:
        st.caption("Shopping behaviour shifts across the months (the holiday season, then early 2020), which makes "
                   "the time-ordered test below a realistic check: the models are tested on months they never saw.")

# --------------------------------------------------------------------------- fair test
st.divider()
left, right = st.columns([1.4, 1], gap="large")
with left:
    st.subheader("A fair test: train on the past, test on the future", anchor=False)
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
        ui.style(fig, height=170, legend=False, left=16, top=10, bottom=40, barmode="overlay")
        fig.update_yaxes(visible=False)
        fig.update_xaxes(type="date", showgrid=False)
        ui.show(fig, key="split")
        ui.legend_items((f"{names[x]}: {sp[x]['first_start'][:10]} → {sp[x]['last_start'][:10]}, "
                         f"{sp[x]['sessions']:,} visits", c) for x, c in zip(("train", "val", "test"),
                                                                             (pal[0], pal[3], pal[1])))
    st.caption("Splitting by time, not at random, is how the engine would really be used: trained on what has "
               "happened, judged on what happens next.")
with right:
    st.subheader("What the models see", anchor=False)
    st.markdown(
        "Each visit is cut after its first **K** clicks (3, 5 or 10). The question: *will this visit end in a "
        "purchase?* Visits that already bought within those clicks are left out, so the answer is never "
        "visible in the input.\n\n"
        "From the K clicks we compute **18 features** in five groups: activity, cart behaviour, the cart/view "
        "ratio, categories, and price & time. The GRU reads the raw clicks instead."
    )
    sk = sorted(summary.get("sessionize", {}).items(), key=lambda kv: int(kv[0]))
    st.dataframe([{"K": int(k_), "Visits usable": v["eligible"], "Buy later": v["positive_rate"]} for k_, v in sk],
                 hide_index=True, width="stretch",
                 column_config={"Visits usable": st.column_config.NumberColumn(format="localized"),
                                "Buy later": st.column_config.NumberColumn(format="percent")})
