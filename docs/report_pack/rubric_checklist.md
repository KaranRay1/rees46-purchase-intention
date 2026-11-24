# Rubric and brief checklist

Status: ✅ evidence exists · ✍️ you must write or do it · ⚠️ gap to close before submission.

## Final-report marking rubric (brief, "Marking Rubric for Final submission")

| Criterion (weight) | What earns High H1 | Evidence in the project | Status |
|---|---|---|---|
| Objectives and motivation (10%) | very challenging objectives, met and discussed | `docs/research_blueprint.md` RQ1–RQ5; `tables/tab_hypotheses` | ✍️ |
| Related work (10%) | papers excellently situate the project | proposal refs [1]–[6] + `references.bib` candidates | ⚠️ `docs/literature_review.md` is empty (TODO): read and verify papers |
| Choice of methods (10%) | selection of complex methods, well-thought-out | 7 models + 10 extensions (`README.md` table) | ✍️ justify each in relation to the RQ |
| Methodology (30%) | hard to fault; all CRISP-DM stages; decisions justified with literature | leakage rules, chronological split, train-only preprocessing, val-only tuning, seeds, Holm | ✍️ cite literature for each key decision |
| Evaluation (20%) | diverse situations, parametrisations, sampling | K ∈ {3,5,10}; 3 seeds; 100% check; drift replay; segments; ablation; efficiency | ✅ |
| Conclusions and future work (5%) | insightful, anchored in literature | facts.md + hypotheses | ✍️ |
| Presentation (15%) | no mistakes; IEEE template; ≤ 7 pages; correct references | `ieee_skeleton/`, `figures/`, `tables/` | ✍️ |

## Brief section 1: "Each project should include"

| # | Item | Where | Status |
|---|---|---|---|
| 1 | Precise RQ and testable hypotheses | blueprint; `tab_hypotheses` | ✅ |
| 2 | Critical literature review and gap | `docs/literature_review.md` (empty) | ⚠️ |
| 3 | Dataset and task definition, preprocessing, limitations | `docs/data_card_full.md`; facts.md *Data* | ✅ |
| 4 | Simple, conventional ML, neural baselines | rule; LR/RF/LightGBM; MLP/TabNet/GRU | ✅ |
| 5 | Splits, hyperparameters, seeds, leakage measures | config, `sessionize.py`, `split.py`, `tests/test_leakage.py` | ✅ |
| 6 | Task-appropriate metrics | PR-AUC (primary), ROC-AUC, F1, precision, recall | ✅ |
| 7 | Efficiency: train time, latency, throughput, size, memory | `rq4_efficiency`; full100 peak memory | ✅ |
| 8 | Ablation / sensitivity / robustness / generalisation | ablation ladder, G6 gap check, 100% run, drift, segments, profit sensitivity | ✅ |
| 9 | **Structured manual** error analysis comparing systems | `error_analysis_sheet.csv` (48 sessions, empty columns) | ⚠️ label it by hand |
| 10 | Limitations, ethics, responsible use, threats to validity | outline §VI notes | ✍️ |
| 11 | Reproducibility package: code, configs, instructions, data access | zip + `HOW_TO_RUN.md` + `00_download_data.py` | ✅ |

## Deliverables (brief section 2)

| Deliverable | Status |
|---|---|
| Final report PDF (IEEE, 5–7 pages) | ✍️ |
| Source code + dataset-access instructions as ZIP | ✅ `bash scripts/make_share_zip.sh` (rebuild at the end) |
| Distribution of work, inside the report | ✍️ agree the wording with Merve |
| 10-minute presentation (PPTX/PDF) + Q&A | ✍️ |
| Name and student number on the front page; AI declaration on the cover sheet | ✍️ `ai_use_log.md` lists the facts |

## Rules to double-check

- [ ] Test set never used for model selection or tuning (it is not: tuning, thresholds and calibration use validation).
- [ ] Changes since the proposal explained (MLOps → efficiency study; merged proposals; extensions; local LLM).
- [ ] Websites (Kaggle, REES46, Ollama) cited as footnotes, not as academic references.
- [ ] Every number in the text matches `facts.md` (ask for a number check once drafted).
- [ ] ≤ 7 pages excluding references and cover sheet.
