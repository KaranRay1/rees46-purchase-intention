"""Data and pipeline: cleaning, eligibility per K, the chronological split, EDA figures."""

import plotly.graph_objects as go
import streamlit as st

import ui

s = ui.store()
summary = s.json("data_summary")

ui.page_header("From 42 million clicks to a fair test",
               "How raw REES46 events become leakage-safe session prefixes with a chronological split.")
if summary is None:
    ui.missing("data summary", "01_prepare_data.py")
    st.stop()

ink = ui.INK[ui.theme()]
clean = summary.get("cleaning")
if not clean:
    st.info("Cleaning counts are missing: step 01 ran with `--no-report`. Rerun it without that flag to show them.")
    clean = {"rows_in": 0, "rows_out": 0, "sessions_out": 0, "purchase_sessions_out": 0, "sources": [], "removed": {}}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw events", ui.fmt_int(clean["rows_in"]), ", ".join(clean["sources"]), delta_color="off")
c2.metric("After cleaning", ui.fmt_int(clean["rows_out"]), f"−{clean['rows_in'] - clean['rows_out']:,} rows",
          delta_color="off")
c3.metric("Sessions", ui.fmt_int(clean["sessions_out"]), f"{clean['purchase_sessions_out']:,} with a purchase",
          delta_color="off")
cut = summary.get("split_cutoffs", {})
c4.metric("Test period starts", cut.get("test_start", "")[:10], "chronological 70 / 15 / 15", delta_color="off")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Rows removed in cleaning", anchor=False)
    rem = {k.replace("_", " "): v for k, v in clean["removed"].items() if v}
    items = sorted(rem.items(), key=lambda kv: kv[1]) or [("nothing removed", 0)]
    fig = go.Figure(go.Bar(
        y=[k for k, _ in items], x=[v for _, v in items], orientation="h",
        marker=dict(color=ui.SERIES[ui.theme()][0], cornerradius=4),
        text=[f"{v:,}" for _, v in items], textposition="outside", cliponaxis=False,
        textfont=dict(color=ink["text"], size=11), hovertemplate="%{y}: %{x:,} rows<extra></extra>",
    ))
    ui.style(fig, height=300, legend=False, left=200)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(title="Rows", range=[0, max(max(v for _, v in items), 1) * 1.25])
    ui.show(fig, key="removed")
    st.caption("The event log contains only view, cart and purchase events: no remove-from-cart, so removal "
               "features were dropped.")

with right:
    st.subheader("Which sessions can be scored at K?", anchor=False)
    sk = sorted(summary.get("sessionize", {}).items(), key=lambda kv: int(kv[0]))
    parts = [("eligible", "Eligible", ui.SERIES[ui.theme()][0]),
             ("purchase_within_prefix", "Bought within first K (excluded)", ui.SERIES[ui.theme()][1]),
             ("too_short", "K events or fewer (excluded)", ink["grid"])]
    fig = go.Figure()
    for key, name, colour in parts:
        fig.add_trace(go.Bar(
            name=name, x=[f"K={k_}" for k_, _ in sk], y=[v[key] for _, v in sk],
            marker=dict(color=colour, line=dict(color=ink["surface"], width=2)),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:,}} sessions<extra></extra>",
        ))
    ui.style(fig, height=320, top=70, barmode="stack")
    fig.update_yaxes(title="Sessions")
    ui.show(fig, key="eligible")
    st.caption("Target: a purchase after event K in the same session. Excluding sessions that already bought "
               "inside the prefix prevents label leakage.")

st.subheader("Chronological split", anchor=False)
rows = []
for k_, sp in sorted(summary.get("splits", {}).items(), key=lambda kv: int(kv[0])):
    for part in ("train", "val", "test"):
        p = sp[part]
        rows.append({"K": int(k_), "Split": part, "Sessions": p["sessions"], "Buyer rate": p["positive_rate"],
                     "From": p["first_start"][:16].replace("T", " "), "To": p["last_start"][:16].replace("T", " ")})
    rows[-1]["Purged at cut-offs"] = sp.get("purged_boundary_sessions")
st.dataframe(rows, hide_index=True, width="stretch", column_config={
    "Sessions": st.column_config.NumberColumn(format="localized"),
    "Buyer rate": st.column_config.NumberColumn(format="percent"),
    "Purged at cut-offs": st.column_config.NumberColumn(format="localized"),
})
st.caption("Cut-offs are shared by every K; sessions that cross a cut-off are dropped so no session spans two splits.")

st.divider()
st.subheader("Exploratory figures", anchor=False)
kind = st.segmented_control("Notebook", ["eda", "feat"], default="eda", key="fig_kind", required=True,
                            format_func={"eda": "01 · EDA", "feat": "02 · Feature analysis"}.get)
figs = s.figures(f"{kind}_")
if not figs:
    st.info("Run the notebooks to produce these figures (see README).")
cols = st.columns(2)
for i, p in enumerate(figs):
    with cols[i % 2]:
        st.image(str(p), caption=p.stem.replace(f"_{s.mode}", "").replace("_", " "), width="stretch")
