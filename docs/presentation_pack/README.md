# Presentation pack

Material for the 10-minute presentation (brief, section 2). Charts and plans only: the slide
text and what you say are yours to write (NCI AI-use rules).

| File | What it is |
|---|---|
| `slide_outline.md` | 11 slides: purpose, chart, numbers to quote (checked against the tables), timing, presenter |
| `qa_prep.md` | likely examiner questions and where the evidence is |
| `charts/` | 9 slide charts, 16:9-ready, PNG (PowerPoint/Keynote/Google Slides) + PDF |

Rebuild the charts after any rerun:

```bash
.venv/bin/python scripts/22_presentation_pack.py --mode full --set data.session_sample_fraction=0.1
```
