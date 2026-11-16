"""Purchase Intent Engine: the shop dashboard and real-time scoring app.

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
    page_title="Purchase Intent Engine",
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
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1360px;}
      /* page header */
      .page-head {display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem;
                  border-bottom: 1px solid rgba(128,128,128,.22); padding-bottom: .9rem; margin-bottom: 1.2rem;}
      .page-head h1 {font-size: 1.75rem; font-weight: 700; letter-spacing: -.02em; margin: .15rem 0 .2rem 0;
                     padding: 0;}
      .page-head p {margin: 0; opacity: .7; font-size: .98rem; max-width: 60rem;}
      .page-meta {font-size: .8rem; opacity: .6; white-space: nowrap; padding-bottom: .2rem;}
      .crumb {font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
              color: var(--primary-color, #2a78d6);}
      @media (max-width: 760px) {.page-head {flex-direction: column; align-items: flex-start;}}
      /* KPI tiles and cards */
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: .7rem;
                                padding: .85rem 1rem; background: rgba(128,128,128,.035);}
      [data-testid="stMetricLabel"] p {font-size: .78rem; opacity: .7; font-weight: 600;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
      .card {border: 1px solid rgba(128,128,128,.22); border-radius: .7rem; padding: 1rem 1.1rem;
             height: 100%; background: rgba(128,128,128,.035);}
      .card h4 {margin: 0 0 .35rem 0; font-size: 1rem; padding: 0;}
      .card p {margin: 0; opacity: .78; font-size: .9rem;}
      .step {font-size: .72rem; font-weight: 700; letter-spacing: .07em; opacity: .6; text-transform: uppercase;}
      h3 {font-size: 1.12rem !important; font-weight: 650 !important;}
      [data-testid="stSidebarNav"] {padding-top: .2rem;}
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

pages = {
    "Business": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("views/data.py", title="Sales & traffic", icon=":material/monitoring:"),
        st.Page("views/products.py", title="Product catalogue", icon=":material/inventory_2:"),
    ],
    "Models": [
        st.Page("views/ml.py", title="Machine learning · Merve", icon=":material/account_tree:"),
        st.Page("views/dl.py", title="Deep learning · Karan", icon=":material/neurology:"),
        st.Page("views/compare.py", title="Model comparison", icon=":material/compare_arrows:"),
    ],
    "Engine": [
        st.Page("views/predict.py", title="Live scoring", icon=":material/bolt:"),
        st.Page("views/planner.py", title="Campaign planner", icon=":material/campaign:"),
        st.Page("views/monitoring.py", title="Monitoring", icon=":material/monitor_heart:"),
    ],
}
st.logo(str(ROOT / "app" / "static" / "logo.svg"), size="large", icon_image=str(ROOT / "app" / "static" / "icon.svg"))
nav = st.navigation(pages)
with st.sidebar:
    st.divider()
    st.selectbox("Dataset", modes, key="mode", format_func=lambda m: ui.MODE_LABELS.get(m, m))
    rq1 = ui.store().csv("rq1_model_comparison")
    ks = ui.available_ks(rq1) or list(ui.cfg()["sessionize"]["prefix_k"])
    if st.session_state.get("k") not in ks:
        st.session_state["k"] = 5 if 5 in ks else ks[0]
    st.segmented_control("Decision point (clicks)", ks, key="k", required=True,
                         help="The models score a visit after its first K clicks. Used by the Models and Engine "
                              "pages.")
    if ui.demo_mode():
        st.caption(":material/cloud: **Online demo:** models and summaries from the bundled results "
                   "(random forest and visit replay not included).")
    st.divider()
    st.caption("Machine learning: Merve Kara  \nDeep learning: Karan Ray  \nNCI · MSc in Artificial Intelligence")
nav.run()
