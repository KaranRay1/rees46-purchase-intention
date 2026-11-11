"""Framework-free serving layer: load trained models and logged results.

``predictor.SessionPredictor`` scores one live session prefix with every saved
model; ``results.ResultStore`` reads the tables the experiment scripts wrote.
Nothing here imports Streamlit, so the same code can sit behind the dashboard
in ``app/`` today and behind an HTTP API (e.g. FastAPI) or a batch job later.
"""
