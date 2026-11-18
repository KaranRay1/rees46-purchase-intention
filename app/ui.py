"""Shared look and helpers for the dashboard pages: palette, Plotly theme, labels, small widgets.

Colours follow the project palette (same fixed slot order as the notebooks and
figures, so a model has the same colour everywhere). Light and dark use their
own validated steps of the same hues.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.config import load_config
from src.serving.results import ResultStore

# --------------------------------------------------------------------------- palette
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb", "neutral": "#f0efec"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "grid": "#383835", "surface": "#1a1a19", "neutral": "#383835"},
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"neg": "#2a78d6", "pos": "#e34948"}      # blue <-> red, grey midpoint
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

MODEL_LABELS = {
    "rule_based": "Rule baseline",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "gru": "GRU (sequence)",
    "transformer": "Transformer (sequence)",
}
MODEL_FAMILY = {
    "rule_based": "Baseline", "logistic_regression": "Baseline",
    "random_forest": "Tree ensemble", "lightgbm": "Tree ensemble",
    "mlp": "Deep tabular", "tabnet": "Deep tabular", "gru": "Sequence", "transformer": "Sequence",
}
GROUP_LABELS = {
    "G1_activity": "G1 Activity",
    "G2_cart_signals": "G2 Cart signals",
    "G3_behavioural_ratios": "G3 Cart/view ratio",
    "G4_category_hierarchy": "G4 Category",
    "G5_price_time": "G5 Price & time",
}
VERDICT_STYLE = {
    "supported": ("✓", "good", "Supported"),
    "partly supported": ("◐", "warning", "Partly supported"),
    "not supported": ("✗", "critical", "Not supported"),
    "exploratory": ("◇", "serious", "Exploratory"),
}


# --------------------------------------------------------------------------- context
def theme() -> str:
    """``"light"`` or ``"dark"``, from the viewer's current Streamlit theme."""
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001 - older Streamlit or no browser context yet
        t = None
    return t if t in ("light", "dark") else "light"


@st.cache_resource(show_spinner=False)
def _config(mode: str) -> dict[str, Any]:
    from src.serving.results import demo_overrides

    cfg = load_config(overrides=[f"data.mode={mode}"])
    demo = demo_overrides(cfg)                 # no trained runs here: use the bundled demo models
    return load_config(overrides=[f"data.mode={mode}", *demo]) if demo else cfg


def demo_mode() -> bool:
    from src.config import resolve_path

    return "demo" in resolve_path(cfg(), "runs_dir").parts


def cfg() -> dict[str, Any]:
    return _config(st.session_state.get("mode", "dev"))


@st.cache_resource(show_spinner=False)
def _store(mode: str) -> ResultStore:
    return ResultStore(_config(mode))


def store() -> ResultStore:
    return _store(st.session_state.get("mode", "dev"))


def k() -> int:
    return int(st.session_state.get("k", 5))


def model_order() -> list[str]:
    return list(cfg()["models"]["enabled"])


def model_colours() -> dict[str, str]:
    pal = SERIES[theme()]
    return {m: pal[i % len(pal)] for i, m in enumerate(model_order())}


def group_colours() -> dict[str, str]:
    pal = SERIES[theme()]
    return {g: pal[i] for i, g in enumerate(GROUP_LABELS)}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# --------------------------------------------------------------------------- plotly
def style(fig: go.Figure, height: int = 380, legend: bool = True, left: int = 64, top: int | None = None,
          bottom: int = 56, **layout: Any) -> go.Figure:
    """Recessive axes and grid, transparent surface, theme-aware ink.

    Margins are explicit (``left`` for long category labels, ``top`` for a legend or top axis):
    Plotly's automargin is not reliable inside Streamlit's iframe-free renderer.
    """
    ink = INK[theme()]
    fig.update_layout(
        height=height,
        margin=dict(l=left, r=24, t=top if top is not None else (56 if legend else 16), b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["text"], size=13),
        hoverlabel=dict(bgcolor=ink["surface"], bordercolor=ink["grid"], font=dict(color=ink["text"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=ink["muted"]),
                    title=None),
        bargap=0.35,
        **layout,
    )
    axis = dict(gridcolor=ink["grid"], zerolinecolor=ink["grid"], linecolor=ink["grid"], automargin=True,
                tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["muted"], size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False, "displayModeBar": "hover"})


# --------------------------------------------------------------------------- widgets
def page_header(title: str, subtitle: str, rq: str | None = None) -> None:
    """Page title with a section breadcrumb and the active dataset on the right."""
    crumb = f"<span class='crumb'>{rq}</span>" if rq else ""
    data = MODE_LABELS.get(st.session_state.get("mode", ""), "")
    st.markdown(
        f"<div class='page-head'><div>{crumb}<h1>{title}</h1><p>{subtitle}</p></div>"
        f"<div class='page-meta'>{'Dataset · ' + data if data else ''}</div></div>",
        unsafe_allow_html=True,
    )


def verdict_badge(verdict: str) -> str:
    icon, status, text = VERDICT_STYLE.get(verdict, ("•", "serious", verdict.capitalize()))
    c = STATUS[status]
    return (f"<span style='display:inline-flex;align-items:center;gap:.35rem;padding:.15rem .6rem;"
            f"border-radius:999px;border:1px solid {c};font-size:.85rem;font-weight:600;white-space:nowrap'>"
            f"<span style='color:{c}'>{icon}</span>{text}</span>")


def missing(what: str, script: str) -> None:
    mode = st.session_state.get("mode", "dev")
    st.info(f"No {what} for mode **{mode}** yet. Produce it with:\n\n"
            f"```bash\npython scripts/{script} --mode {mode}\n```", icon=":material/hourglass_empty:")


def to_pandas_display(df: pl.DataFrame, rename: Mapping[str, str] | None = None) -> Any:
    out = df.rename({k: v for k, v in (rename or {}).items() if k in df.columns})
    return out.to_pandas()


def fmt_money(n: float | int) -> str:
    return "$" + fmt_int(n)


def month_name(m: str, short: bool = False) -> str:
    """``"2019-10"`` -> ``"Oct 2019"`` (``"Oct 19"`` when short)."""
    from datetime import datetime

    return datetime.strptime(m, "%Y-%m").strftime("%b %y" if short else "%b %Y")


@st.cache_resource(show_spinner="Summarising sales per day…")
def _daily(mode: str) -> pl.DataFrame | None:
    from src.serving.catalog import daily_profile

    return daily_profile(_config(mode))


@st.cache_resource(show_spinner="Building the product catalogue (first run only)…")
def _product_month(mode: str) -> pl.DataFrame | None:
    from src.serving.catalog import product_month

    return product_month(_config(mode))


def daily() -> pl.DataFrame | None:
    return _daily(st.session_state.get("mode", "dev"))


def product_month() -> pl.DataFrame | None:
    return _product_month(st.session_state.get("mode", "dev"))


def sample_note() -> str | None:
    """A one-line caveat when the mode's events are a sample of visits, else None."""
    clean = (store().json("data_summary") or {}).get("cleaning") or {}
    if clean.get("removed", {}).get("session_sampling"):
        share = clean["sessions_out"] / max(clean["sessions_in"], 1)
        return (f"Counts and revenue are from a random {share:.0%} sample of visits; rates (conversion, cart rate) "
                "represent the whole shop.")
    return None


def fmt_int(n: float | int) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}".replace(".00", "")
    return f"{n:,.0f}"


def models_in(df: pl.DataFrame, column: str = "model") -> list[str]:
    present = set(df[column].to_list())
    return [m for m in model_order() if m in present]


def per_k(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("k") == k()) if "k" in df.columns else df


def available_ks(*tables: pl.DataFrame | None) -> list[int]:
    ks: set[int] = set()
    for t in tables:
        if t is not None and "k" in t.columns:
            ks |= set(t["k"].to_list())
    return sorted(ks)


def legend_items(items: Iterable[tuple[str, str]]) -> None:
    """A compact HTML legend (label, colour) for charts drawn as separate traces."""
    html = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.35rem;margin-right:1rem'>"
        f"<span style='width:10px;height:10px;border-radius:3px;background:{c};display:inline-block'></span>"
        f"<span style='opacity:.8'>{t}</span></span>" for t, c in items
    )
    st.markdown(f"<div style='font-size:.85rem'>{html}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- teams, modes, plain words
TEAMS = {
    "ml": {
        "title": "Machine learning", "author": "Merve Kara", "icon": ":material/account_tree:",
        "models": ["rule_based", "logistic_regression", "random_forest", "lightgbm"],
        "ablation_model": "lightgbm",
        "pitch": "Classic models on 18 hand-built session features: fast, light and easy to explain.",
    },
    "dl": {
        "title": "Deep learning", "author": "Karan Ray", "icon": ":material/neurology:",
        "models": ["mlp", "tabnet", "gru"],
        "ablation_model": "mlp",
        "pitch": "Neural networks: two on the same features, one (GRU) reading the raw click sequence.",
    },
}
MODEL_INFO = {
    "rule_based": "Says “buyer” if the shopper added anything to the cart. The bar to beat.",
    "logistic_regression": "A weighted sum of the features: the standard linear reference.",
    "random_forest": "Hundreds of decision trees that vote.",
    "lightgbm": "Gradient-boosted trees, each fixing the previous trees' mistakes.",
    "mlp": "A feed-forward neural network on the same 18 features.",
    "tabnet": "A neural network that picks which features to attend to at each step.",
    "gru": "A recurrent network that reads the clicks one by one, in order.",
}
MODE_LABELS = {
    "full": "Oct 2019 – Apr 2020 (7 months)",
    "dev": "Oct 2019 (1 month)",
    "synthetic": "Synthetic test data",
}


def team_of(model: str) -> str:
    return next((t for t, v in TEAMS.items() if model in v["models"]), "ml")


def team_colours() -> dict[str, str]:
    pal = SERIES[theme()]
    return {"ml": pal[0], "dl": pal[1]}


def author_chip(team: str) -> None:
    t, c = TEAMS[team], team_colours()[team]
    st.markdown(
        f"<span style='display:inline-flex;align-items:center;gap:.45rem;padding:.2rem .75rem;border-radius:999px;"
        f"border:1px solid {c};font-size:.85rem;font-weight:600'><span style='width:9px;height:9px;border-radius:50%;"
        f"background:{c};display:inline-block'></span>{t['title']} · built &amp; explained by {t['author']}</span>",
        unsafe_allow_html=True,
    )


def card(title: str, body: str, kicker: str = "") -> str:
    k_ = f"<div class='step'>{kicker}</div>" if kicker else ""
    return f"<div class='card'>{k_}<h4>{title}</h4><p>{body}</p></div>"
