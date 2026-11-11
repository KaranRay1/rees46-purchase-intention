"""Purchase-intention research dashboard.

Run from the project root::

    streamlit run app/main.py

The sidebar picks the data mode (``synthetic`` / ``dev`` / ``full``: whichever
have results in ``results/tables``) and the prefix length K; every page reads
the tables written by ``scripts/01``–``06`` for that mode, and the live
predictor loads the models those runs saved. Nothing is retrained here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Purchase Intention · REES46",
    page_icon=":material/shopping_cart_checkout:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import load_config, resolve_path  # noqa: E402
from src.serving.results import modes_with_results  # noqa: E402

import ui  # noqa: E402

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: .75rem;
                                padding: .8rem 1rem; background: rgba(128,128,128,.04);}
      [data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .75;}
      h1 {letter-spacing: -.02em;}
      .card {border: 1px solid rgba(128,128,128,.22); border-radius: .75rem; padding: 1rem 1.1rem;
             height: 100%; background: rgba(128,128,128,.04);}
      .card h4 {margin: 0 0 .35rem 0; font-size: 1rem;}
      .card p {margin: 0; opacity: .78; font-size: .9rem;}
      .step {font-size: .75rem; font-weight: 700; letter-spacing: .06em; opacity: .6;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- sidebar
tables_dir = resolve_path(load_config(), "tables_dir")
modes = modes_with_results(tables_dir) or ["dev"]
with st.sidebar:
    st.markdown("### :material/shopping_cart_checkout: Purchase intention")
    st.caption("Session-based prediction on REES46 · tabular ML vs deep models")
    mode_help = {"dev": "October 2019 (one month)", "full": "Oct 2019 – Apr 2020", "synthetic": "Generated test data"}
    st.selectbox("Data", modes, key="mode", format_func=lambda m: f"{m} · {mode_help.get(m, m)}")
    rq1 = ui.store().csv("rq1_model_comparison")
    ks = ui.available_ks(rq1) or list(ui.cfg()["sessionize"]["prefix_k"])
    if st.session_state.get("k") not in ks:
        st.session_state["k"] = 5 if 5 in ks else ks[0]
    st.segmented_control("Prefix length K (events seen)", ks, key="k", required=True,
                         help="Predictions are made after the first K events of a session.")

pages = {
    "Study": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("views/data.py", title="Data & pipeline", icon=":material/database:"),
    ],
    "Results": [
        st.Page("views/models.py", title="RQ1–2 · Models & K", icon=":material/leaderboard:"),
        st.Page("views/features.py", title="RQ3 · Features", icon=":material/stacked_bar_chart:"),
        st.Page("views/efficiency.py", title="RQ4 · Real-time cost", icon=":material/speed:"),
        st.Page("views/explain.py", title="RQ5 · Explainability", icon=":material/psychology:"),
    ],
    "Try it": [
        st.Page("views/predict.py", title="Live predictor", icon=":material/bolt:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.divider()
    st.caption(f"Results: `{tables_dir.relative_to(ROOT) if tables_dir.is_relative_to(ROOT) else tables_dir}`  \n"
               "NCI MSc AI · Machine Learning")
nav.run()
