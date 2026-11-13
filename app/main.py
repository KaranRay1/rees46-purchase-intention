"""Purchase-intention research dashboard.

Run from the project root::

    streamlit run app/main.py

The sidebar picks the data (``full`` = Oct 2019 – Apr 2020, ``dev`` = Oct 2019;
whichever have results in ``results/tables``) and the prefix length K; every page reads
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
real = [m for m in modes if m != "synthetic"]
modes = real or modes                                   # synthetic data only when nothing real exists
done = [m for m in modes if (tables_dir / f"rq1_model_comparison_{m}.csv").exists()]
modes = done + [m for m in modes if m not in done]      # modes with model results first (the default)
if st.session_state.get("mode") not in modes:
    st.session_state["mode"] = modes[0]
with st.sidebar:
    st.markdown("### :material/shopping_cart_checkout: Purchase Intent Engine")
    st.caption("Will this shopper buy? Answered from the first few clicks, in milliseconds.")
    st.selectbox("Data", modes, key="mode", format_func=lambda m: ui.MODE_LABELS.get(m, m))
    rq1 = ui.store().csv("rq1_model_comparison")
    ks = ui.available_ks(rq1) or list(ui.cfg()["sessionize"]["prefix_k"])
    if st.session_state.get("k") not in ks:
        st.session_state["k"] = 5 if 5 in ks else ks[0]
    st.segmented_control("Clicks seen before predicting (K)", ks, key="k", required=True,
                         help="The models decide after the first K events of a visit.")

pages = {
    "1 · Research": [
        st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("views/data.py", title="The data", icon=":material/database:"),
    ],
    "2 · Analysis": [
        st.Page("views/ml.py", title="Machine learning · Merve", icon=":material/account_tree:"),
        st.Page("views/dl.py", title="Deep learning · Karan", icon=":material/neurology:"),
        st.Page("views/compare.py", title="ML vs DL", icon=":material/compare_arrows:"),
    ],
    "3 · Product": [
        st.Page("views/predict.py", title="Live demo", icon=":material/bolt:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.divider()
    st.caption("Karan Ray · Merve Kara  \nNCI MSc in AI · Machine Learning")
nav.run()
