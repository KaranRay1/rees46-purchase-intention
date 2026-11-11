"""Live predictor: build a session (or replay a real test session) and score it with every saved model."""

from __future__ import annotations

import zlib
from datetime import datetime, time, timedelta, timezone

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from src.config import processed_dir_for_k, resolve_path
from src.serving.predictor import SessionPredictor, available_runs
from src.utils.notebook import prepared_data_status

K, mode = ui.k(), st.session_state.get("mode", "dev")
ink = ui.INK[ui.theme()]
colours = ui.model_colours()

ui.page_header("Score a session live",
               f"Describe the first {K} events of a visit, or replay a real one from the test week. Every model "
               "trained in step 03 scores it, using exactly the training feature pipeline.")


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

st.divider()
yes = [sc for sc in scores if sc.decision]
learned = [sc for sc in scores if sc.model != "rule_based"]
c1, c2, c3 = st.columns(3)
c1.metric("Models predicting a purchase", f"{len(yes)} / {len(scores)}")
best = next((sc for sc in scores if sc.model == "lightgbm"), learned[0] if learned else scores[0])
c2.metric(f"{ui.label(best.model)} score", f"{best.score:.0%}", f"threshold {best.threshold:.0%}", delta_color="off")
c3.metric("Slowest model, end to end", f"{max(sc.latency_ms for sc in scores):.1f} ms",
          "feature transform + inference in this app", delta_color="off",
          help="Includes Python-side preprocessing for a single session. Pure inference latency is benchmarked on the RQ4 page.")

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("Each model's verdict", anchor=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[ui.label(sc.model) for sc in scores], x=[sc.score for sc in scores], orientation="h",
        marker=dict(color=[colours.get(sc.model, ink["muted"]) for sc in scores], cornerradius=4,
                    opacity=[1 if sc.decision else 0.45 for sc in scores]),
        text=[f"{sc.score:.0%} · {'buyer' if sc.decision else 'no'}" for sc in scores], textposition="outside",
        cliponaxis=False, textfont=dict(color=ink["text"], size=11), showlegend=False,
        customdata=[[sc.threshold, sc.latency_ms] for sc in scores],
        hovertemplate="<b>%{y}</b><br>score %{x:.3f} · threshold %{customdata[0]:.3f}"
                      "<br>%{customdata[1]:.1f} ms<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=[ui.label(sc.model) for sc in scores], x=[sc.threshold for sc in scores], mode="markers",
        name="Decision threshold", marker=dict(symbol="line-ns", size=22, line=dict(width=2.5, color=ink["text"])),
        hovertemplate="threshold %{x:.3f}<extra></extra>",
    ))
    ui.style(fig, height=360, left=150, top=40)
    fig.update_xaxes(range=[0, 1.28], tickformat=".0%", title="Purchase score", tickvals=[0, .25, .5, .75, 1])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    ui.show(fig, key="verdicts")
    st.caption("Solid bars cross their model's threshold (tuned on the validation week). The rule baseline "
               "says “buyer” whenever the prefix contains a cart.")

with right:
    st.subheader("Why LightGBM thinks so", anchor=False)
    if contrib is None:
        st.info("LightGBM is not among the trained models for this K.")
    else:
        body = contrib.filter(pl.col("feature") != "(base)")
        top = body.head(8)
        rest = body.slice(8)["contribution"].sum()
        names = [f"{f} = {v}" if v else f for f, v in zip(top["feature"], top["value"])]
        vals = top["contribution"].to_list()
        if body.height > 8:
            names.append(f"{body.height - 8} other features")
            vals.append(rest)
        fig = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker=dict(color=[ui.DIVERGING["pos"] if v > 0 else ui.DIVERGING["neg"] for v in vals], cornerradius=4),
            text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=11),
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        ))
        ui.style(fig, height=360, legend=False, left=280)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        span = max(abs(v) for v in vals) if vals else 1.0
        fig.update_xaxes(title="Log-odds (TreeSHAP)", zeroline=True, zerolinecolor=ink["muted"],
                         range=[min(0.0, min(vals)) - span * 0.45, max(0.0, max(vals)) + span * 0.4])
        ui.show(fig, key="contrib")
        ui.legend_items([("Pushes toward a purchase", ui.DIVERGING["pos"]), ("Pushes away", ui.DIVERGING["neg"])])

with st.expander("The 18 features the models saw"):
    st.dataframe(feats.select(pl.all().cast(pl.String))
                 .transpose(include_header=True, header_name="feature", column_names=["value"]),
                 hide_index=True, width="stretch")
with st.expander("Models used"):
    st.dataframe([{"Model": ui.label(sc.model), "Run": sc.run_id, "Threshold": round(sc.threshold, 4),
                   "Latency ms": round(sc.latency_ms, 2)} for sc in scores], hide_index=True, width="stretch")
