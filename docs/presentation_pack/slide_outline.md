# Presentation plan: 10 minutes + Q&A (notes, not slide text)

> Same rule as the report: write every slide title, bullet and spoken sentence yourselves. This
> plan says **what each slide must show, which chart to use, which numbers to quote** (with
> sources) and how long to spend. Numbers: `../report_pack/facts.md`.

Format: in person, 10 minutes, then Q&A; upload PPTX or PDF to Moodle (brief, section 2).
Two presenters: suggested split below (agree it with Merve).

| # | Slide (purpose) | Chart (`charts/`) | Numbers to have on it | Time | Who |
|---|---|---|---|---|---|
| 1 | Title: project title, both names and student numbers, module | – | – | 0:15 | either |
| 2 | Problem and research question: why predict purchase in the first clicks; RQ and hypotheses H1–H5 | – | buyer rate ≈ 10% (imbalance) | 1:00 | Karan |
| 3 | Data and leakage-safe setup: REES46, 7 months, sessions → eligible → split; prefix-only features | `s_data_funnel` | 89.69 M → 8.87 M → 1.90 M (K=5) → 1.32 M train / 296 k test | 1:15 | Merve |
| 4 | Methods: rule baseline; LR, RF, LightGBM (ML); MLP, TabNet, GRU (DL); 3 seeds; paired bootstrap + Holm; PR-AUC | – (a simple two-column diagram you draw) | seeds 42/7/2026 | 1:00 | both (own models) |
| 5 | Main result (RQ1/RQ2): ML ≈ DL; GRU vs RF tie; lift falls with K | `s_fig1_rq1_model_comparison` (+ `s_fig2` optional) | K=5: RF 0.3369, GRU 0.3358, Holm p 0.36; 100% data: GRU +0.0013 significant | 1:30 | Karan |
| 6 | What drives predictions (RQ3/RQ5): cart block dominates; ratios add nothing | `s_fig3_rq3_feature_ladder` | G1 → G1+G2 jump; SHAP share G2 ≈ 0.43 | 1:00 | Merve |
| 7 | Efficiency (RQ4): engine choice | `s_fig4_rq4_accuracy_vs_latency` | p95: GRU 0.0875 ms vs RF 5.41 ms; RF 105 MB vs GRU 0.07 MB | 0:45 | Karan |
| 8 | What actually helps: extensions in one view; user history is the big one | `s_fig5_extensions_forest` + `s_user_history` | user history +0.074 / +0.055 / +0.039 (K=3/5/10); 24 h gap check +0.036 | 1:15 | Karan |
| 9 | Robustness and fairness: 100% run, drift replay, segments | `s_fig6_segments_lift` | lift ratio price 0.54; new vs returning flag rate 13.3% vs 12.9%; GRU vs RF differ in 3/72 segments | 0:45 | Merve |
| 10 | Error analysis: your manual categories (from the worksheet) | a table you build from `error_analysis_sheet.csv` | counts per category per group | 0:45 | Merve |
| 11 | Limitations, ethics, conclusion, future work | – | – | 0:45 | both |
| (backup) | LLM agent vs RAG (only if time / asked) | `s_fig7_agent_vs_rag` | tool Qs: agent 100% (Llama) vs RAG 33% | – | Karan |
| (backup) | Business value (if asked) | – | +$3.86 M/month vs offer-to-all at K=5 (illustrative) | – | Karan |

Total ≈ 10:15: rehearse with a timer and trim slide 6 or 9 if over.

## Live demo (optional, risky in 10 minutes)
The dashboard (`streamlit run app/main.py`) runs offline from the demo bundle. If you show it,
use 30–45 seconds on **Live scoring** only, and keep a screenshot slide as a fallback. Start
Ollama beforehand if you show the assistant (backup only).

## Design tips
- One message per slide; the chart is the slide, the title states the takeaway (in your words).
- Charts are 16:9-ready PNG (and PDF). Use the same blue = ML, orange = DL colours as the charts.
- Put the numbers you will say aloud on the slide, large; everything else goes in the report.
