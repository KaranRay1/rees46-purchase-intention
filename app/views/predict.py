"""Live demo of the purchase-intent engine: clicks in, purchase probability and a next action out."""

from __future__ import annotations

import zlib
from datetime import datetime, time, timedelta, timezone

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui
from src.config import processed_dir_for_k, resolve_path
from src.serving.predictor import SessionPredictor, available_runs
from src.utils.notebook import prepared_data_status

K, mode = ui.k(), st.session_state.get("mode", "dev")
ink = ui.INK[ui.theme()]

ui.page_header("Try the engine",
               f"This is how a shop would use it: send the first {K} clicks of a visit, get back a purchase "
               "probability and a suggested action. Build a visit yourself or replay a real one from the test period.",
               rq="3 · Product")


@st.cache_resource(show_spinner="Loading the trained models…")
def predictor(mode: str, k: int) -> SessionPredictor:
    return SessionPredictor(ui._config(mode), k)


@st.cache_data(show_spinner=False)
def test_sessions(mode: str, k: int) -> pl.DataFrame:
    """Test-split sessions with their true outcome (from the primary LightGBM run, else any model)."""
    cfg = ui._config(mode)
    runs = available_runs(cfg).filter(pl.col("k") == k)
    pick = runs.filter(pl.col("model") == "lightgbm")
    run_id = (pick if pick.height else runs)["run_id"][0]
    preds = pl.read_parquet(resolve_path(cfg, "runs_dir") / run_id / "predictions.parquet")
    return preds.filter(pl.col("split") == "test").select("user_session", "y_true")


@st.cache_data(show_spinner=False)
def session_events(mode: str, k: int, session: str) -> pl.DataFrame:
    path = processed_dir_for_k(ui._config(mode), k) / "prefix_events.parquet"
    return (pl.scan_parquet(path).filter(pl.col("user_session") == session).sort("pos").collect()
            .with_columns(pl.col("event_type").cast(pl.String)))


try:
    model = predictor(mode, K)
except FileNotFoundError as exc:
    st.warning(str(exc), icon=":material/model_training:")
    st.stop()

# --------------------------------------------------------------------------- input
CATEGORIES = ["electronics.smartphone", "electronics.audio.headphone", "electronics.video.tv", "computers.notebook",
              "appliances.kitchen.washer", "appliances.environment.vacuum", "apparel.shoes", "furniture.bedroom.bed",
              "auto.accessories.player", "kids.toys", "unknown"]
PRESETS = {
    "Decided buyer": [("view", "electronics.smartphone", "apple", 1013.0, 0, 1), ("view", "electronics.smartphone", "apple", 1013.0, 25, 1),
                      ("cart", "electronics.smartphone", "apple", 1013.0, 40, 1), ("view", "electronics.audio.headphone", "apple", 179.0, 60, 2),
                      ("cart", "electronics.audio.headphone", "apple", 179.0, 20, 2)],
    "Comparison shopper": [("view", "electronics.smartphone", "samsung", 245.0, 0, 1), ("view", "electronics.smartphone", "xiaomi", 210.0, 35, 2),
                           ("view", "electronics.smartphone", "samsung", 260.0, 40, 3), ("view", "electronics.smartphone", "huawei", 230.0, 30, 4),
                           ("cart", "electronics.smartphone", "xiaomi", 210.0, 50, 2)],
    "Window shopper": [("view", "furniture.bedroom.bed", "unknown", 450.0, 0, 1), ("view", "apparel.shoes", "nike", 95.0, 120, 2),
                       ("view", "kids.toys", "lego", 40.0, 200, 3), ("view", "electronics.video.tv", "lg", 600.0, 90, 4),
                       ("view", "computers.notebook", "acer", 520.0, 300, 5)],
}


def preset_frame(name: str) -> pl.DataFrame:
    rows = list(PRESETS[name][:K])
    last = rows[-1]
    rows += [("view", last[1], last[2], last[3], 30, last[5])] * (K - len(rows))  # longer K: keep browsing
    return pl.DataFrame(rows, schema=["event_type", "category_code", "brand", "price", "seconds_after_previous",
                                      "product"], orient="row")


def to_events(table: pl.DataFrame, start: datetime) -> pl.DataFrame:
    secs = table["seconds_after_previous"].fill_null(0).cast(pl.Float64).to_list()
    t, times = start, []
    for i, sec in enumerate(secs):
        t = t + timedelta(seconds=0 if i == 0 else max(sec, 0))
        times.append(t)
    return pl.DataFrame({
        "event_time": times,
        "event_type": table["event_type"],
        "product_id": table["product"].fill_null(0).cast(pl.Int64),
        "category_id": [zlib.crc32((c or "unknown").encode()) for c in table["category_code"]],
        "category_code": table["category_code"].fill_null("unknown"),
        "brand": table["brand"].fill_null("unknown"),
        "price": table["price"].cast(pl.Float64),
    })


build, replay = st.tabs([":material/edit: Build a session", ":material/replay: Replay a real test session"])
with build:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    preset = c1.selectbox("Start from", list(PRESETS), key=f"preset_{K}")
    day = c2.date_input("Day", datetime(2019, 10, 28).date(), key="day")
    hour = c3.time_input("Time (UTC)", time(14, 0), key="hour", step=900)
    table = st.data_editor(
        preset_frame(preset).to_pandas(), key=f"editor_{K}_{preset}", hide_index=True, width="stretch",
        num_rows="fixed",
        column_config={
            "event_type": st.column_config.SelectboxColumn("Event", options=["view", "cart"], required=True),
            "category_code": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
            "brand": st.column_config.TextColumn("Brand"),
            "price": st.column_config.NumberColumn("Price", min_value=0.01, format="%.2f", required=True),
            "seconds_after_previous": st.column_config.NumberColumn("Seconds after previous", min_value=0, step=1),
            "product": st.column_config.NumberColumn("Product #", min_value=1, step=1,
                                                     help="Same number = the same product viewed again."),
        },
    )
    built = to_events(pl.from_pandas(table), datetime.combine(day, hour, tzinfo=timezone.utc))
    st.caption("Purchases cannot appear in the prefix: the task is to predict a purchase *after* these events.")

with replay:
    status = prepared_data_status(ui._config(mode)).get(K)
    if status != mode:
        st.info(f"The processed data in `data/processed/` is for mode **{status or 'none'}**, not **{mode}**. "
                f"Run `python scripts/01_prepare_data.py --mode {mode}` to replay real sessions.")
    else:
        sessions = test_sessions(mode, K)
        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button("Random buyer", icon=":material/shopping_bag:", width="stretch"):
            st.session_state[f"replay_{K}"] = sessions.filter(pl.col("y_true") == 1).sample(1)["user_session"][0]
        if b2.button("Random non-buyer", icon=":material/directions_walk:", width="stretch"):
            st.session_state[f"replay_{K}"] = sessions.filter(pl.col("y_true") == 0).sample(1)["user_session"][0]
        b3.caption(f"{sessions.height:,} test sessions · {sessions['y_true'].mean():.1%} went on to buy")
        sid = st.session_state.get(f"replay_{K}")
        if sid:
            ev = session_events(mode, K, sid)
            truth = int(sessions.filter(pl.col("user_session") == sid)["y_true"][0])
            st.markdown(f"Session `{sid[:13]}…`, actual outcome after event {K}: "
                        + (ui.verdict_badge("supported").replace("Supported", "Bought")
                           if truth else ui.verdict_badge("not supported").replace("Not supported", "Did not buy")),
                        unsafe_allow_html=True)
            st.dataframe(ev.select("pos", "event_time", "event_type", "category_code", "brand", "price"),
                         hide_index=True, width="stretch")

use_replay = bool(st.session_state.get(f"replay_{K}")) and st.toggle(
    "Score the replayed session (off: score the session you built)", value=True, key="use_replay")
events = session_events(mode, K, st.session_state[f"replay_{K}"]) if use_replay else built

# --------------------------------------------------------------------------- predict
try:
    scores = model.predict(events)
    feats = model.features(events)
    contrib = model.contributions(events)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

by_model = {sc.model: sc for sc in scores}
engines = [m for m in by_model if m != "rule_based"] or list(by_model)
rq1, eff = ui.store().csv("rq1_model_comparison"), ui.store().csv("rq4_efficiency")
if rq1 is not None:
    ship, _ = team_page.recommended(rq1, ui.store().csv("rq1_pairwise_tests"), eff, K)
    if ship in engines:
        engines.remove(ship)
        engines.insert(0, ship)

st.divider()
head, pick = st.columns([3, 1.2])
head.subheader("The engine's answer", anchor=False)
engine = pick.selectbox("Engine model", engines, format_func=lambda m: ("★ " if m == engines[0] and rq1 is not None else "")
                        + f"{ui.label(m)} · {ui.team_of(m).upper()}",
                        help="★ = recommended: the fastest of the models statistically tied for the best accuracy.",
                        key="engine")
sc = by_model[engine]
if sc.decision:
    level, colour, action = ("High intent", ui.STATUS["good"],
                             "Keep the path clear: express checkout, no discount needed.")
elif sc.score >= sc.threshold / 2:
    level, colour, action = ("Warm", ui.STATUS["warning"],
                             "Give a nudge: free shipping, a low-stock note or a reminder of the cart.")
else:
    level, colour, action = ("Just browsing", ui.STATUS["serious"],
                             "Don't spend a coupon here; show inspiration and similar products instead.")
icon = {"High intent": "●", "Warm": "◐", "Just browsing": "○"}[level]
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Purchase probability", f"{sc.score:.0%}", f"buyer threshold {sc.threshold:.0%}", delta_color="off", delta_arrow="off",
          help="The threshold was tuned on the validation period to balance precision and recall.")
c2.markdown(f"<div style='padding:.4rem 0'><div style='opacity:.7;font-size:.82rem'>Intent</div>"
            f"<div style='font-size:1.6rem;font-weight:700'><span style='color:{colour}'>{icon}</span> {level}</div>"
            f"<div style='opacity:.65;font-size:.82rem'>answered in {sc.latency_ms:.1f} ms</div></div>",
            unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='step'>SUGGESTED ACTION</div><p style='opacity:1;font-size:1rem'>"
            f"{action}</p></div>", unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("Why · LightGBM explanation", anchor=False)
    if contrib is None:
        st.info("LightGBM is not among the trained models for this K.")
    else:
        body = contrib.filter(pl.col("feature") != "(base)")
        top = body.head(6)
        rest = body.slice(6)["contribution"].sum()
        names = [f"{f} = {v}" if v else f for f, v in zip(top["feature"], top["value"])]
        vals = top["contribution"].to_list()
        if body.height > 6:
            names.append(f"{body.height - 6} other features")
            vals.append(rest)
        fig = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker=dict(color=[ui.DIVERGING["pos"] if v > 0 else ui.DIVERGING["neg"] for v in vals], cornerradius=4),
            text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        ))
        ui.style(fig, height=330, legend=False, left=230)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        span = max(abs(v) for v in vals) if vals else 1.0
        fig.update_xaxes(title="Effect (log-odds)", zeroline=True, zerolinecolor=ink["muted"],
                         range=[min(0.0, min(vals)) - span * 0.45, max(0.0, max(vals)) + span * 0.4])
        ui.show(fig, key="contrib")
        ui.legend_items([("Points to a purchase", ui.DIVERGING["pos"]), ("Points away", ui.DIVERGING["neg"])])

with right:
    st.subheader("Second opinions from every model", anchor=False)
    tcol = ui.team_colours()
    order = [m for t in ("ml", "dl") for m in ui.TEAMS[t]["models"] if m in by_model]
    fig = go.Figure()
    for team in ("ml", "dl"):
        ms = [m for m in order if ui.team_of(m) == team]
        fig.add_trace(go.Bar(
            y=[ui.label(m) for m in ms], x=[by_model[m].score for m in ms], orientation="h",
            name=f"{ui.TEAMS[team]['title']} · {ui.TEAMS[team]['author']}",
            marker=dict(color=tcol[team], cornerradius=4, opacity=[1 if by_model[m].decision else 0.4 for m in ms]),
            text=[f"{by_model[m].score:.0%} · {'buyer' if by_model[m].decision else 'no'}" for m in ms],
            textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        y=[ui.label(m) for m in order], x=[by_model[m].threshold for m in order], mode="markers",
        name="Threshold", marker=dict(symbol="line-ns", size=20, line=dict(width=2.5, color=ink["text"])),
        hovertemplate="threshold %{x:.1%}<extra></extra>",
    ))
    ui.style(fig, height=360, left=150, top=50, barmode="overlay")
    fig.update_xaxes(range=[0, 1.3], tickformat=".0%", title="Purchase probability", tickvals=[0, .25, .5, .75, 1])
    fig.update_yaxes(categoryorder="array", categoryarray=[ui.label(m) for m in order][::-1], showgrid=False)
    ui.show(fig, key="verdicts")
    yes = sum(by_model[m].decision for m in order)
    st.caption(f"{yes} of {len(order)} models call this visit a buyer. Faded bars stay below their threshold.")

with st.expander("Use the engine from your own code"):
    st.code(f"""from src.config import load_config
from src.serving.predictor import SessionPredictor

engine = SessionPredictor(load_config(overrides=["data.mode={mode}"]), k={K}, models=["{engine}"])
result = engine.predict(events)[0]   # events: the visit's first {K} clicks (polars DataFrame)
print(result.score, result.decision, result.latency_ms)""", language="python")
    st.caption("The same function can sit behind a web API; it loads the saved model once and scores a visit in "
               "milliseconds.")
with st.expander(f"The {feats.width} features computed from these clicks"):
    st.dataframe(feats.select(pl.all().cast(pl.String))
                 .transpose(include_header=True, header_name="feature", column_names=["value"]),
                 hide_index=True, width="stretch")
