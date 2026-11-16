"""One page per team: Machine learning (Merve Kara) and Deep learning (Karan Ray).

Both pages show the same four questions for their own models, so they can be
presented side by side: what was built, how well it predicts, what it looks at,
and how fast it is. Only results that hold up statistically are stated as findings.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui

from src.serving.recommend import ALPHA, recommended, significant_vs  # noqa: F401  (re-exported for the pages)


def render(team: str) -> None:
    t = ui.TEAMS[team]
    other = "dl" if team == "ml" else "ml"
    s, K = ui.store(), ui.k()
    rq1 = s.csv("rq1_model_comparison")
    pairs = s.csv("rq1_pairwise_tests")
    eff = s.csv("rq4_efficiency")
    groups = s.csv(f"rq5_importance_groups_k{K}")
    abl = s.csv("rq3_ablation")
    steps = s.csv("rq3_step_tests")
    ink = ui.INK[ui.theme()]
    colours = ui.model_colours()

    ui.page_header(f"{t['title']} models", t["pitch"], rq="Models")
    ui.author_chip(team)
    if rq1 is None:
        ui.missing("model results", "03_run_experiments.py")
        st.stop()
    models = [m for m in t["models"] if m in set(rq1["model"].to_list())]
    part = ui.per_k(rq1)

    # ------------------------------------------------------------------ what was built
    st.subheader("What was built", anchor=False)
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        col.markdown(ui.card(ui.label(m), ui.MODEL_INFO.get(m, "")), unsafe_allow_html=True)

    # ------------------------------------------------------------------ accuracy
    st.write("")
    st.subheader(f"How well do they spot buyers? · after {K} clicks", anchor=False)
    mine = part.filter(pl.col("model").is_in(models)).sort("pr_auc_mean")
    theirs = part.filter(pl.col("model").is_in(ui.TEAMS[other]["models"])).sort("pr_auc_mean", descending=True)
    base = float(part["base_rate"][0])
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        fig = go.Figure(go.Bar(
            y=[ui.label(m) for m in mine["model"]], x=mine["pr_auc_mean"], orientation="h",
            marker=dict(color=[colours[m] for m in mine["model"]], cornerradius=4),
            error_x=dict(type="data", symmetric=False, array=mine["pr_auc_ci_high"] - mine["pr_auc_mean"],
                         arrayminus=mine["pr_auc_mean"] - mine["pr_auc_ci_low"], color=ink["muted"], thickness=1.5,
                         width=4),
            text=[f"{v:.3f}  ({lift:.1f}× random)" for v, lift in zip(mine["pr_auc_mean"], mine["pr_auc_lift_mean"])],
            textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=12),
            hovertemplate="<b>%{y}</b><br>PR-AUC %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=base, line=dict(color=ink["muted"], width=1.5, dash="dot"))
        ref = ""
        if theirs.height:
            b = theirs.row(0, named=True)
            fig.add_vline(x=b["pr_auc_mean"], line=dict(color=ui.team_colours()[other], width=2, dash="dash"))
            ref = (f" Dashed line: the best {ui.TEAMS[other]['title'].lower()} model, {ui.label(b['model'])} "
                   f"({b['pr_auc_mean']:.3f}).")
        hi = float(part["pr_auc_ci_high"].max())
        ui.style(fig, height=80 + 62 * len(models), legend=False, left=150, top=16)
        fig.update_xaxes(range=[0, hi * 1.45], title="PR-AUC (higher is better)")
        fig.update_yaxes(showgrid=False)
        ui.show(fig, key=f"{team}_acc")
        st.caption(f"Dotted line: random guessing ({base:.3f}).{ref}")
    with right:
        st.markdown(
            "**How to read it.** PR-AUC measures how well a model ranks the shoppers who *will* buy above those "
            f"who won't. Guessing at random scores the buyer rate (**{base:.1%}** here); 1.0 would be perfect. "
            "Whiskers: 95 % confidence interval."
        )
        findings = []
        for m in models:
            if m in ("rule_based", "logistic_regression"):
                continue
            for ref in ("logistic_regression", "rule_based"):
                r = significant_vs(pairs, K, m, ref)
                if r and r[1] and r[0] > 0:
                    findings.append(f"**{ui.label(m)}** beats {ui.label(ref).lower()} by **{r[0]:+.3f}**.")
                    break
        if findings:
            st.markdown("**Confirmed by significance tests**")
            for f in findings:
                st.markdown(f"- {f}")

    # ------------------------------------------------------------------ what they look at
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What the models look at", anchor=False)
        expl = [m for m in models if groups is not None and m in groups.columns]
        if not expl:
            st.caption("No attribution results for these models at this K.")
        else:
            gcol = ui.group_colours()
            fig = go.Figure()
            for g, name in ui.GROUP_LABELS.items():
                row = groups.filter(pl.col("feature") == g)
                if not row.height:
                    continue
                vals = [float(row[m][0]) for m in expl]
                fig.add_trace(go.Bar(
                    name=name, y=[ui.label(m) for m in expl], x=vals, orientation="h",
                    marker=dict(color=gcol[g], line=dict(color=ink["surface"], width=2)),
                    text=[f"{v:.0%}" if v >= 0.1 else "" for v in vals], textposition="inside",
                    insidetextanchor="middle", textfont=dict(color="#ffffff", size=11),
                    hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.0%}}<extra></extra>",
                ))
            ui.style(fig, height=70 + 55 * len(expl), legend=False, left=150, top=16, barmode="stack")
            fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of the model's attention")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            ui.show(fig, key=f"{team}_groups")
            ui.legend_items((name, gcol[g]) for g, name in ui.GROUP_LABELS.items())
            note = "SHAP values" + (", TabNet's own attention masks" if "tabnet" in expl else "")
            st.caption(f"Measured with {note}. The rule baseline and the GRU are not split by feature group.")
    with right:
        am = t["ablation_model"]
        st.subheader(f"What each signal adds · {ui.label(am)}", anchor=False)
        a = None if abl is None else ui.per_k(abl).filter((pl.col("model") == am) & ~pl.col("is_variant")).sort("step")
        if a is None or not a.height:
            st.caption("No feature-ladder results for this K.")
        else:
            sig = {}
            if steps is not None:
                st_ = ui.per_k(steps).filter((pl.col("model") == am) & (pl.col("family") != "variant"))
                sig = {r["to"]: r["p_bootstrap_holm"] < ALPHA and r["pr_auc_gain"] > 0 for r in st_.iter_rows(named=True)}
            short = ["Activity", "Cart", "Cart/view<br>ratio", "Category", "Price &<br>time"]
            names = short[: a.height]
            gains = [a["pr_auc_mean"][0]] + [float(x) for x in a["pr_auc_gain"][1:]]
            ok = [True] + [sig.get(fs, False) for fs in a["feature_set"][1:]]
            gcol = ui.group_colours()
            fig = go.Figure(go.Bar(
                x=names, y=gains,
                marker=dict(color=[gcol[g] if o else ink["grid"] for g, o in zip(ui.GROUP_LABELS, ok)], cornerradius=4),
                text=[f"{gains[0]:.3f}"] + [f"+{v:.3f}" if o else "" for v, o in zip(gains[1:], ok[1:])],
                textposition="outside", cliponaxis=False, textfont=dict(color=ink["text"], size=12),
                hovertemplate="%{x}<br>%{y:+.4f} PR-AUC<extra></extra>",
            ))
            for x, o in zip(names[1:], ok[1:]):
                if not o:
                    fig.add_annotation(x=x, y=0, text="≈0", showarrow=False, yshift=12,
                                       font=dict(color=ink["muted"], size=12))
            ui.style(fig, height=330, legend=False, left=56, bottom=64)
            fig.update_yaxes(title="PR-AUC added", range=[0, max(gains) * 1.25])
            fig.update_xaxes(showgrid=False, tickangle=0, tickfont=dict(size=11))
            ui.show(fig, key=f"{team}_ladder")
            st.caption("Starting from activity counts alone (first bar), feature groups are added one at a time, "
                       "left to right. “≈0”: the gain is not statistically significant, so we do not claim it.")

    # ------------------------------------------------------------------ speed
    st.divider()
    st.subheader("Fast enough for a live shop?", anchor=False)
    if eff is None:
        ui.missing("speed benchmarks", "05_benchmark_latency.py")
        return
    e = ui.per_k(eff).filter(pl.col("model").is_in(models))
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        r = e.filter(pl.col("model") == m)
        if r.height:
            ms = float(r["latency_b1_p95_ms"][0])
            col.metric(ui.label(m), f"{ms:.2f} ms" if ms >= 0.01 else "< 0.01 ms",
                       f"{r['model_size_mb'][0]:.1f} MB model", delta_color="off", delta_arrow="off",
                       help="95th-percentile time to score one session on one CPU thread (model only).")
    st.caption("Every model here answers well inside a 10 ms real-time budget.")
