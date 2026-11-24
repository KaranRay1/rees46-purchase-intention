# Report evidence pack

Material to **write the final report from**. It contains evidence and planning aids, not report
text: the NCI brief does not allow AI-written submission content (see `ai_use_log.md`).

| File | What it is |
|---|---|
| `outline.md` | each required section: what the brief asks, the points the evidence supports, where the numbers are, page budget |
| `facts.md` | every headline number with its source table (auto-generated) |
| `figures/` | 7 paper figures, IEEE column/page width, PDF (for the paper) + PNG (preview) |
| `tables/` | 5 key tables as LaTeX (booktabs) + CSV |
| `error_analysis_sheet.csv` | 48 test sessions for the **manual** error analysis: fill in `error_type`, `likely_cause`, `notes` |
| `rubric_checklist.md` | brief and marking rubric mapped to the evidence; open gaps marked ⚠️ |
| `references.bib` | the proposal's references + candidates to read and verify |
| `ieee_skeleton/main.tex` | empty IEEE conference skeleton: headings, figure slots, page budgets |
| `ai_use_log.md` | facts for the AI-use declaration on the cover sheet |

Rebuild figures, tables, facts and the worksheet after any rerun:

```bash
.venv/bin/python scripts/21_report_pack.py --mode full --set data.session_sample_fraction=0.1
```

(Rebuilding regenerates `error_analysis_sheet.csv`: copy your filled-in sheet elsewhere first.)

**Word instead of LaTeX?** Use the IEEE conference Word template from the IEEE website and insert the
PNG/PDF figures; the CSV tables paste into Word tables.
