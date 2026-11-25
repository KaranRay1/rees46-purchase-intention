"""Step 24: empty skeletons for the presentation (PowerPoint) and the report (Word).

The NCI brief does not allow AI-written report or slide content, so these files hold only
structure and evidence: the layout, the charts and tables, the project-flow diagram, model
names, and numbers copied from the result tables. Every title, sentence and caption is a
``[WRITE …]`` box that the authors fill in themselves.

Needs ``python-pptx`` and ``python-docx`` (not in requirements.txt: ``pip install python-pptx
python-docx``). Run steps 21 and 22 first (figures and slide charts).

Writes ``docs/presentation_pack/presentation_skeleton.pptx`` and
``docs/report_pack/report_skeleton.docx``.

Examples::

    python scripts/24_report_slide_skeletons.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches as DIn
from docx.shared import Pt as DPt
from docx.shared import RGBColor as DRGB
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "docs" / "presentation_pack" / "charts"
FIGS = ROOT / "docs" / "report_pack" / "figures"
TABLES = ROOT / "docs" / "report_pack" / "tables"

INK, NAVY, MUTED = RGBColor(0x1D, 0x26, 0x33), RGBColor(0x0B, 0x25, 0x45), RGBColor(0x5C, 0x67, 0x73)
ML, DL, GOOD = RGBColor(0x00, 0x72, 0xB2), RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73)
GREY, WRITE_BG, WRITE_LINE = RGBColor(0x8A, 0x94, 0x9E), RGBColor(0xF3, 0xF5, 0xF8), RGBColor(0x9A, 0xA5, 0xB1)
WHITE, PALE = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xE9, 0xF1, 0xF8)
FONT = "Calibri"
W, H = 13.333, 7.5


# --------------------------------------------------------------------------- slides
def _run(par, text, size=16, bold=False, italic=False, color=INK):
    r = par.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = color
    return r


def text(slide, x, y, w, h, lines, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         italic=False, name="Text"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, line in enumerate([lines] if isinstance(lines, str) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(4)
        _run(p, line, size, bold, italic, color)
    return tb


def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Box", dash=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb, s.line.width = line, Pt(1.25)
        if dash:
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def write(slide, x, y, w, h, prompt, size=15, dark=False):
    """A dashed box the authors replace with their own words."""
    b = box(slide, x, y, w, h, NAVY if dark else WRITE_BG, RGBColor(0x7F, 0x93, 0xAD) if dark else WRITE_LINE,
            name="WRITE", dash=True)
    tf = b.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    _run(p, "[WRITE] ", size, True, False, RGBColor(0xC9, 0xD6, 0xE6) if dark else MUTED)
    _run(p, prompt, size, False, True, RGBColor(0xC9, 0xD6, 0xE6) if dark else GREY)
    return b


def frame(prs, kicker, title_prompt, notes, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY if dark else WHITE
    text(s, 0.6, 0.35, 9, 0.35, kicker.upper(), 12, True, RGBColor(0x8F, 0xB8, 0xE0) if dark else ML, name="Kicker")
    write(s, 0.6, 0.75, 12.1, 0.8, title_prompt, 22, dark)
    s.notes_slide.notes_text_frame.text = notes
    return s


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    pic.name = path.stem
    return pic


def fit(slide, path, x, y, w, h):
    """Place an image as large as possible inside the box, centred."""
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return picture(slide, path, x + (w - pw) / 2, y + (h - ph) / 2, pw, ph)


def stat(slide, x, y, w, value, label, color=NAVY, size=36):
    text(slide, x, y, w, 0.7, value, size, True, color, name="Stat")
    text(slide, x, y + 0.72, w, 0.6, label, 13, False, MUTED, name="StatLabel")


def chip(slide, x, y, w, h, label, fill, color=WHITE, size=13):
    b = box(slide, x, y, w, h, fill, name="Chip")
    tf = b.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, label, size, True, False, color)
    return b


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb, c.line.width = GREY, Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = OxmlElement("a:tailEnd")
    tail.set("type", "triangle")
    ln.append(tail)
    return c


def deck(out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 1 title (dark)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    text(s, 0.8, 1.6, 11.7, 1.8, "Real-Time Session-Based E-Commerce Purchase Intention Prediction", 40, True, WHITE,
         name="Title")
    text(s, 0.8, 3.55, 11.7, 0.5, "Predicting a purchase from the first K clicks of a visit (REES46, 7 months)", 20,
         False, RGBColor(0xC9, 0xD6, 0xE6), name="Subtitle")
    chip(s, 0.8, 4.6, 3.6, 0.55, "Karan Ray · deep learning", DL, size=15)
    chip(s, 4.6, 4.6, 3.9, 0.55, "Merve Kara · machine learning", ML, size=15)
    text(s, 0.8, 5.4, 11, 0.5, "[ADD student numbers] · MSc in Artificial Intelligence · Machine Learning · "
         "National College of Ireland", 15, False, RGBColor(0xC9, 0xD6, 0xE6), name="Meta")
    s.notes_slide.notes_text_frame.text = "[WRITE your opening line]. About 15 s."

    # 2 problem statement
    s = frame(prs, "Problem statement", "your title: the problem in one line",
              "[WRITE your script]. Numbers to quote: 89.69 M raw sessions in 7 months (data_summary); "
              "buyer rate 9.64% of eligible visits at K=5 (10.16% in the test weeks). About 1:00.")
    write(s, 0.6, 1.85, 6.6, 2.4, "the problem: why a shop wants to know, after only a few clicks, whether a visit "
          "will end in a purchase, and why this is hard")
    write(s, 0.6, 4.45, 6.6, 1.55, "the main research question, in your words")
    for i, (v, lab) in enumerate([("89.7 M", "visits (sessions) in 7 months of REES46 clickstream"),
                                  ("≈ 10 %", "of eligible visits buy: an imbalanced problem"),
                                  ("3 · 5 · 10", "clicks seen before predicting (K)")]):
        box(s, 7.7, 1.85 + i * 1.42, 5.0, 1.27, PALE, name="StatCard")
        stat(s, 7.95, 1.92 + i * 1.42, 4.6, v, lab, NAVY, 30)
    for i, (rq, lab) in enumerate([("RQ1", "model families"), ("RQ2", "how early (K)"), ("RQ3", "which features"),
                                   ("RQ4", "efficiency"), ("RQ5", "explainability")]):
        chip(s, 0.6 + i * 1.34, 6.25, 1.24, 0.7, f"{rq}\n{lab}", NAVY, size=11)

    # 3 data
    s = frame(prs, "Data", "your title: what the data is and how it was reduced",
              "[WRITE your script]. Numbers: 411.7 M events / 89.69 M sessions raw; 10% session sample -> "
              "8.87 M sessions; K=5 eligible 1.90 M; train 1.32 M, validation 285 k, test 296 k (chronological "
              "70/15/15; test = 4-30 Apr 2020). About 1:15.")
    fit(s, CHARTS / "s_data_funnel.png", 0.6, 1.8, 8.3, 3.6)
    write(s, 9.2, 1.8, 3.5, 3.6, "leakage-safe setup: only the first K clicks are used, visits that buy inside them "
          "are removed, split by time", 14)
    for i, (lab, c) in enumerate([("1 Oct 2019 – 1 Mar 2020\ntrain 70 %", ML), ("1 Mar – 4 Apr 2020\nvalidation 15 %", GREY),
                                  ("4 – 30 Apr 2020\ntest 15 %", DL)]):
        chip(s, 0.6 + i * 4.1, 5.85, 3.9, 0.95, lab, c, size=14)

    # 4 methods: algorithms
    s = frame(prs, "Methods · algorithms", "your title: which models and why",
              "[WRITE your script]. Each of you presents your own models. Settings: config/config.yaml. About 1:00.")
    cols = [("Baseline", GREY, [("Rule", "cart in the first K clicks → buyer")]),
            ("Machine learning · Merve", ML, [("Logistic regression", "linear reference · scikit-learn"),
                                              ("Random forest", "bagged trees · scikit-learn"),
                                              ("LightGBM", "gradient-boosted trees")]),
            ("Deep learning · Karan", DL, [("MLP", "feed-forward net · PyTorch"),
                                           ("TabNet", "attentive tabular DL"),
                                           ("GRU", "recurrent net over the clicks · PyTorch")])]
    xs, ws = [0.6, 3.75, 8.3], [2.9, 4.3, 4.4]
    for (head, c, models), x, w in zip(cols, xs, ws):
        chip(s, x, 1.8, w, 0.55, head, c, size=15)
        for j, (name, desc) in enumerate(models):
            box(s, x, 2.5 + j * 0.95, w, 0.82, PALE, name="ModelCard")
            text(s, x + 0.15, 2.53 + j * 0.95, w - 0.3, 0.4, name, 16, True, INK, name="Model")
            text(s, x + 0.15, 2.9 + j * 0.95, w - 0.3, 0.4, desc, 12, False, MUTED, name="ModelDesc")
    write(s, 0.6, 5.5, 12.1, 1.3, "why these models: simple → conventional ML → neural; aggregated features vs the "
          "raw click sequence (GRU)", 14)

    # 5 methods: evaluation + extensions
    s = frame(prs, "Methods · evaluation and extensions", "your title: how you made the comparison fair",
              "[WRITE your script]. PR-AUC primary because ~10% buyers; 1000 bootstrap resamples; Holm alpha 0.05; "
              "threshold, tuning and calibration on validation only. About 0:45 (can merge with slide 4).")
    text(s, 0.6, 1.8, 6, 0.4, "Evaluation", 18, True, NAVY, name="Head")
    for i, lab in enumerate(["PR-AUC (primary)", "3 seeds · 42 / 7 / 2026", "paired bootstrap CIs",
                             "McNemar test", "Holm correction", "test weeks never used for tuning"]):
        chip(s, 0.6 + (i % 2) * 3.0, 2.35 + (i // 2) * 0.75, 2.85, 0.6, lab, NAVY, size=13)
    text(s, 6.9, 1.8, 5.8, 0.4, "Extensions", 18, True, NAVY, name="Head")
    for i, lab in enumerate(["User history (G6)", "Embeddings", "Optuna tuning", "Transformer", "Stacking",
                             "Calibration", "Conformal sets", "Drift & retraining", "Profit model", "Segments",
                             "RAG assistant", "Tool-using agent"]):
        chip(s, 6.9 + (i % 3) * 1.97, 2.35 + (i // 3) * 0.75, 1.87, 0.6, lab, GOOD, size=12)
    write(s, 0.6, 5.5, 12.1, 1.3, "why PR-AUC and paired tests; which extensions you will talk about", 14)

    # 6 project flow
    s = frame(prs, "Project flow", "your title: how the project runs from raw data to the dashboard",
              "[WRITE your script]. Each box is a script in scripts/ (numbers in brackets). About 0:45.")
    steps = [("Download", "00"), ("Clean &\nsessionise", "01"), ("Features\nG1–G5", "02"), ("Train 7 models\n× 3 K × 3 seeds", "03"),
             ("Ablation", "04"), ("Latency", "05"), ("SHAP", "06")]
    bw, gap, y0 = 1.55, 0.2, 2.0
    for i, (lab, n) in enumerate(steps):
        x = 0.6 + i * (bw + gap)
        chip(s, x, y0, bw, 1.05, lab, ML if i < 3 else NAVY, size=13)
        text(s, x, y0 + 1.1, bw, 0.3, f"step {n}", 11, False, MUTED, PP_ALIGN.CENTER, name="StepNo")
        if i < len(steps) - 1:
            arrow(s, x + bw + 0.02, y0 + 0.52, x + bw + gap - 0.02, y0 + 0.52)
    ext = [("Extensions", "10–20"), ("Report &\nslide packs", "21–23"), ("Dashboard\n& API", "app/")]
    for i, (lab, n) in enumerate(ext):
        x = 0.6 + 4 * (bw + gap) + i * (bw + gap) - (bw + gap)
        chip(s, x, 3.95, bw, 1.05, lab, GOOD if i == 0 else GREY, size=13)
        text(s, x, 5.05, bw, 0.3, n, 11, False, MUTED, PP_ALIGN.CENTER, name="StepNo")
    cx = 0.6 + 3 * (bw + gap) + bw / 2                 # trained models feed the extensions
    arrow(s, cx, y0 + 1.42, cx, 3.93)
    for i in range(2):
        x = 0.6 + (3 + i) * (bw + gap) + bw
        arrow(s, x + 0.02, 4.47, x + gap - 0.02, 4.47)
    write(s, 0.6, 5.6, 12.1, 1.2, "the flow in your words: what each stage produces and why the order matters", 14)

    # 7 result: model comparison
    s = frame(prs, "Results · RQ1 model comparison", "your title: the main result",
              "[WRITE your script]. K=5 PR-AUC: RF 0.3369, GRU 0.3358 (Holm p 0.36: tie), LightGBM 0.3331, "
              "MLP 0.3327, TabNet 0.3265, LR 0.3250, rule 0.2144. On 100% of sessions GRU 0.3371 vs RF 0.3358, "
              "difference significant. About 1:30.")
    fit(s, CHARTS / "s_fig1_rq1_model_comparison.png", 0.6, 1.75, 9.0, 4.0)
    for i, (v, lab, c) in enumerate([("0.337", "Random forest PR-AUC (K=5)", ML), ("0.336", "GRU PR-AUC (K=5)", DL),
                                     ("p = 0.36", "RF vs GRU after Holm: no significant difference", NAVY)]):
        stat(s, 9.9, 1.8 + i * 1.35, 2.9, v, lab, c, 30)
    write(s, 0.6, 5.95, 12.1, 0.9, "what this means: ML vs DL", 14)

    # 8 result: features
    s = frame(prs, "Results · RQ3/RQ5 what drives the prediction", "your title: which behaviour matters",
              "[WRITE your script]. LightGBM K=5 ladder: G1 0.2254 -> +G2 0.3180 (+0.0915, significant) -> +G3 "
              "0.3178 (no gain) -> +G4 0.3287 -> +G5 0.3331. Mean SHAP share: G2 cart 0.43, G5 price/time 0.30. "
              "About 1:00.")
    fit(s, CHARTS / "s_fig3_rq3_feature_ladder.png", 0.6, 1.75, 7.4, 4.9)
    for i, (v, lab, c) in enumerate([("+0.092", "PR-AUC from adding the cart block (G2)", GOOD),
                                     ("≈ 0", "added by behavioural ratios (G3)", GREY),
                                     ("43 %", "of SHAP importance is cart signals", NAVY)]):
        stat(s, 8.4, 1.8 + i * 1.25, 4.3, v, lab, c, 30)
    write(s, 8.4, 5.6, 4.3, 1.2, "your interpretation", 14)

    # 9 result: efficiency
    s = frame(prs, "Results · RQ4 efficiency", "your title: is it fast enough, and which engine",
              "[WRITE your script]. p95 latency per visit (batch 1): GRU 0.0875 ms, LightGBM 0.158 ms, RF 5.41 ms; "
              "size GRU 0.07 MB vs RF 105 MB. Budget assumed 10 ms. About 0:45.")
    fit(s, CHARTS / "s_fig4_rq4_accuracy_vs_latency.png", 0.6, 1.75, 8.2, 4.6)
    for i, (v, lab, c) in enumerate([("0.09 ms", "GRU p95 latency per visit", DL), ("5.4 ms", "Random forest p95", ML),
                                     ("≈ 1,500×", "smaller model: GRU 0.07 MB vs RF 105 MB", NAVY)]):
        stat(s, 9.2, 1.8 + i * 1.35, 3.6, v, lab, c, 30)
    write(s, 0.6, 6.0, 12.1, 0.85, "the engine you chose and why", 14)

    # 10 result: extensions
    s = frame(prs, "Results · extensions", "your title: what actually helps",
              "[WRITE your script]. K=5 gains (paired bootstrap, Holm): user history +0.0555 (significant); "
              "embeddings +0.0019; tuning GRU +0.0036; stacking +0.0012 (n.s.); Transformer vs GRU +0.0012 (n.s.). "
              "User history +0.074 / +0.055 / +0.039 at K=3/5/10. About 1:15.")
    fit(s, CHARTS / "s_fig5_extensions_forest.png", 0.6, 1.75, 8.6, 4.1)
    stat(s, 9.6, 1.85, 3.2, "+0.056", "PR-AUC from user history (G6) at K=5, the largest gain", GOOD, 32)
    stat(s, 9.6, 3.35, 3.2, "< 0.004", "every other extension (tuning, Transformer, stacking, embeddings)", GREY, 32)
    write(s, 0.6, 6.0, 12.1, 0.85, "your takeaway about the extensions", 14)

    # 11 robustness and fairness
    s = frame(prs, "Results · robustness and fairness", "your title: does it hold up",
              "[WRITE your script]. 100% of sessions: GRU 0.3371, RF 0.3358 (conclusions hold). Segments (GRU, "
              "K=5): lowest lift ratio = price level; new vs returning users flagged 13.3% vs 12.9%; GRU and RF "
              "differ in 3 of 72 segments. Drift replay: retraining helps a little (10 of 12 significant). About 0:45.")
    fit(s, CHARTS / "s_fig6_segments_lift.png", 0.6, 1.75, 6.4, 5.1)
    for i, (v, lab, c) in enumerate([("10 % → 100 %", "PR-AUC moves ≤ 0.0014 on all 89.7 M sessions", NAVY),
                                     ("13.3 % vs 12.9 %", "new vs returning users flagged as likely buyers", GOOD),
                                     ("3 of 72", "segments where GRU and RF differ", GREY)]):
        stat(s, 7.4, 1.8 + i * 1.3, 5.3, v, lab, c, 28)
    write(s, 7.4, 5.75, 5.3, 1.05, "your interpretation and limits", 14)

    # 12 main points (dark)
    s = frame(prs, "Main points", "your title for the conclusion",
              "[WRITE your script]. Four points, one sentence each, in your own words. About 0:45.", dark=True)
    facts = [("ML ≈ DL", "RF 0.337 vs GRU 0.336 at K=5"), ("Cart signal", "+0.092 PR-AUC, 43 % of SHAP"),
             ("Real time", "GRU 0.09 ms per visit"), ("User history", "+0.056 PR-AUC, the largest gain")]
    for i, (h, f) in enumerate(facts):
        x = 0.6 + i * 3.1
        box(s, x, 1.85, 2.9, 4.3, RGBColor(0x14, 0x36, 0x5E), name="PointCard")
        text(s, x + 0.2, 1.95, 1, 0.6, f"{i + 1}", 30, True, RGBColor(0x8F, 0xB8, 0xE0), name="No")
        text(s, x + 0.2, 2.6, 2.5, 0.5, h, 20, True, WHITE, name="Point")
        text(s, x + 0.2, 3.15, 2.5, 0.7, f, 13, False, RGBColor(0xC9, 0xD6, 0xE6), name="Fact")
        write(s, x + 0.15, 4.05, 2.6, 1.9, "one-sentence takeaway", 13, dark=True)
    write(s, 0.6, 6.35, 12.1, 0.6, "limitations and future work in one line", 13, dark=True)

    # 13 questions (dark)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    text(s, 0.8, 2.6, 11.7, 1.2, "Questions", 48, True, WHITE, name="Title")
    text(s, 0.8, 3.8, 11.7, 0.6, "Code, results and dashboard: [ADD repository link or 'available on request']", 16,
         False, RGBColor(0xC9, 0xD6, 0xE6), name="Meta")
    s.notes_slide.notes_text_frame.text = "Q&A preparation: docs/presentation_pack/qa_prep.md"

    # backups
    s = frame(prs, "Backup · user history", "your title (backup slide)",
              "Backup if asked. User history gain K=3/5/10: +0.074 / +0.055 / +0.039; the 24 h gap check keeps "
              "+0.036 (not leakage).")
    fit(s, CHARTS / "s_user_history.png", 0.6, 1.8, 12.1, 4.6)
    s = frame(prs, "Backup · LLM agent vs RAG", "your title (backup slide)",
              "Backup if asked. Tool questions (n=18): agent 100% (Llama 3.1 8B) vs RAG 33%; standard questions "
              "(n=30): agent 82% vs RAG 93%.")
    fit(s, CHARTS / "s_fig7_agent_vs_rag.png", 0.6, 1.8, 8.0, 5.0)
    write(s, 8.9, 1.8, 3.8, 5.0, "what the agent adds over plain retrieval, and where it still fails", 14)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)


# --------------------------------------------------------------------------- report
def _cols(section, n):
    sp = section._sectPr
    cols = sp.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sp.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), "360")


def _para(doc, txt="", size=10, bold=False, italic=False, color=None, align=None, after=4):
    p = doc.add_paragraph()
    if txt:
        r = p.add_run(txt)
        r.font.size, r.bold, r.italic = DPt(size), bold, italic
        if color:
            r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = DPt(after)
    return p


def _write(doc, prompt, words=None):
    p = doc.add_paragraph()
    r = p.add_run("[WRITE] ")
    r.bold, r.font.size, r.font.color.rgb = True, DPt(9.5), DRGB(0x5C, 0x67, 0x73)
    r = p.add_run(prompt + (f" (≈ {words} words)" if words else ""))
    r.italic, r.font.size, r.font.color.rgb = True, DPt(9.5), DRGB(0x8A, 0x94, 0x9E)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F5F8")
    p._p.get_or_add_pPr().append(shd)
    p.paragraph_format.space_after = DPt(6)
    return p


def _evidence(doc, txt):
    p = doc.add_paragraph()
    r = p.add_run("Evidence: " + txt)
    r.font.size, r.font.color.rgb = DPt(8), DRGB(0x00, 0x72, 0xB2)
    p.paragraph_format.space_after = DPt(6)


def _heading(doc, txt):
    p = doc.add_paragraph()
    r = p.add_run(txt.upper())
    r.font.size = DPt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.space_after = DPt(8), DPt(4)


def _figure(doc, path, width, label):
    doc.add_picture(str(path), width=DIn(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    r = p.add_run(f"{label}. ")
    r.font.size = DPt(8)
    r = p.add_run("[WRITE caption]")
    r.italic, r.font.size, r.font.color.rgb = True, DPt(8), DRGB(0x8A, 0x94, 0x9E)


def _table(doc, name, label, fmt=None):
    rows = list(csv.reader((TABLES / f"{name}.csv").open()))
    p = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    r = p.add_run(f"{label}. ")
    r.font.size = DPt(8)
    r = p.add_run("[WRITE caption]")
    r.italic, r.font.size, r.font.color.rgb = True, DPt(8), DRGB(0x8A, 0x94, 0x9E)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            v = v.replace("$\\pm$", "±")
            if fmt and i > 0 and j in fmt:
                v = fmt[j](v)
            cell = t.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(v)
            run.font.size, run.bold = DPt(7), i == 0
    _para(doc, after=6)


def report(out: Path) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", DPt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    sec = doc.sections[0]
    sec.page_width, sec.page_height = DIn(8.27), DIn(11.69)
    sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin = DIn(0.75), DIn(1.0), DIn(0.63), DIn(0.63)

    _para(doc, "Real-Time Session-Based E-Commerce Purchase Intention Prediction", 22,
          align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    _para(doc, "Karan Ray [ADD student number]  ·  Merve Kara [ADD student number]", 11,
          align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    _para(doc, "MSc in Artificial Intelligence, National College of Ireland", 10, italic=True,
          align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    _para(doc, "Skeleton only: replace every [WRITE] box with your own text, delete the blue 'Evidence' lines, and "
          "keep 5–7 pages. For the final file the official IEEE conference template is safest; this skeleton "
          "mirrors its layout. Evidence map: docs/report_pack/outline.md; numbers: docs/report_pack/facts.md.", 8,
          italic=True, color=DRGB(0xD5, 0x5E, 0x00), align=WD_ALIGN_PARAGRAPH.CENTER, after=6)

    two = doc.add_section(WD_SECTION.CONTINUOUS)
    _cols(two, 2)
    p = _para(doc, after=4)
    r = p.add_run("Abstract—")
    r.bold, r.italic, r.font.size = True, True, DPt(9)
    _write(doc, "problem, data, methods, main result with numbers, conclusion", "100–150")
    p = _para(doc, after=8)
    r = p.add_run("Index Terms—")
    r.bold, r.italic, r.font.size = True, True, DPt(9)
    _write(doc, "4–6 keywords")

    _heading(doc, "I. Introduction")
    _write(doc, "motivation and problem statement; research question and RQ1–RQ5; contributions", "350")
    _evidence(doc, "outline.md §I; docs/research_blueprint.md §2 (RQs, hypotheses)")
    _heading(doc, "II. Related Work")
    _write(doc, "session-based purchase prediction, tabular ML vs DL, sequence models; the gap you address", "450")
    _evidence(doc, "references.bib (verify every candidate by reading it); your literature review")
    _heading(doc, "III. Methods and Baselines")
    _write(doc, "rule baseline; LR, RF, LightGBM (Merve); MLP, TabNet, GRU (Karan); extensions you discuss; "
           "why each, strengths, limits, compute", "350")
    _evidence(doc, "project_inventory.pdf §3; tab_main_k5 (train time, size)")
    _heading(doc, "IV. Methodology and Experimental Setup")
    _write(doc, "data and cleaning; sessions, K and target; leakage rules; chronological split; features G1–G5 (+G6); "
           "preprocessing; hyperparameters and tuning on validation; seeds; statistics", "550")
    _evidence(doc, "facts.md Data; config/config.yaml; tab_data_splits")
    _table(doc, "tab_data_splits", "TABLE I", {3: lambda v: f"{float(v):.2%}", 2: lambda v: f"{int(v):,}"})
    _heading(doc, "V. Results and Evaluation")
    _write(doc, "A. main comparison (RQ1) and how early (RQ2)", "300")

    full = doc.add_section(WD_SECTION.CONTINUOUS)
    _cols(full, 1)
    _figure(doc, FIGS / "fig1_rq1_model_comparison.png", 7.0, "Fig. 1")
    two = doc.add_section(WD_SECTION.CONTINUOUS)
    _cols(two, 2)
    _table(doc, "tab_main_k5", "TABLE II")
    _write(doc, "B. efficiency (RQ4) and the engine choice", "200")
    _figure(doc, FIGS / "fig4_rq4_accuracy_vs_latency.png", 3.4, "Fig. 2")
    _write(doc, "C. feature ablation and SHAP (RQ3, RQ5)", "250")
    _figure(doc, FIGS / "fig3_rq3_feature_ladder.png", 3.4, "Fig. 3")
    _write(doc, "D. tuning and extensions (user history, embeddings, Transformer, stacking)", "250")
    _figure(doc, FIGS / "fig5_extensions_forest.png", 3.4, "Fig. 4")
    _write(doc, "E. robustness: 100% of sessions, drift replay, segments", "250")
    _table(doc, "tab_full_scale", "TABLE III")
    _figure(doc, FIGS / "fig6_segments_lift.png", 3.4, "Fig. 5 (optional)")
    _write(doc, "F. manual error analysis: your categories and counts from error_analysis_sheet.csv", "200")
    _evidence(doc, "outline.md §V; facts.md; error_analysis_sheet.csv (label it by hand first)")
    _heading(doc, "VI. Limitations, Ethics and Responsible Use")
    _write(doc, "validity threats, bias and segments, privacy of user history, deployment risks, LLM use", "250")
    _heading(doc, "VII. Conclusion and Future Work")
    _write(doc, "answer each RQ (see Table IV), contribution, realistic future work", "150")
    _table(doc, "tab_hypotheses", "TABLE IV (optional)")
    _heading(doc, "Reproducibility Statement")
    _write(doc, "code, data access, environment, config, seeds, procedure", "80")
    _evidence(doc, "HOW_TO_RUN.md; scripts/00_download_data.py; requirements.txt; config/config.yaml")
    _heading(doc, "Distribution of Work")
    _write(doc, "who did what, agreed between you", "80")
    _heading(doc, "References")
    _write(doc, "IEEE style, only works you have read")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)


def main() -> None:
    pptx_out = ROOT / "docs" / "presentation_pack" / "presentation_skeleton.pptx"
    docx_out = ROOT / "docs" / "report_pack" / "report_skeleton.docx"
    deck(pptx_out)
    report(docx_out)
    print(f"-> {pptx_out.relative_to(ROOT)}\n-> {docx_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
