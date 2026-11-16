"""Machine learning vs deep learning: who wins, by how much, at what cost, and which model to ship."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import team_page
import ui

s, K = ui.store(), ui.k()
rq1 = s.csv("rq1_model_comparison")
pairs = s.csv("rq1_pairwise_tests")
eff = s.csv("rq4_efficiency")
summary = s.json("rq5_explain_summary")

ui.page_header("Model comparison",
               "Machine learning vs deep learning on the same visits, test period and metric, and which model "
               "the engine runs.",
               rq="Models")
if rq1 is None:
    ui.missing("model results", "03_run_experiments.py")
    st.stop()

ink = ui.INK[ui.theme()]
tcol = ui.team_colours()
part = ui.per_k(rq1)
learned = part.filter(pl.col("model") != "rule_based")


def best(team: str, df: pl.DataFrame = part) -> dict | None:
    d = df.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]) & (pl.col("model") != "rule_based"))
    return d.sort("pr_auc_mean", descending=True).row(0, named=True) if d.height else None


bm, bd = best("ml"), best("dl")

# --------------------------------------------------------------------------- verdict
if bm and bd:
    test = team_page.significant_vs(pairs, K, bm["model"], bd["model"])
    diff = bm["pr_auc_mean"] - bd["pr_auc_mean"]
    winner = "ml" if diff > 0 else "dl"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Best {ui.TEAMS['ml']['title'].lower()} · {ui.TEAMS['ml']['author']}", ui.label(bm["model"]),
              f"PR-AUC {bm['pr_auc_mean']:.3f}", delta_color="off", delta_arrow="off")
    c2.metric(f"Best {ui.TEAMS['dl']['title'].lower()} · {ui.TEAMS['dl']['author']}", ui.label(bd["model"]),
              f"PR-AUC {bd['pr_auc_mean']:.3f}", delta_color="off", delta_arrow="off")
    if test and test[1]:
        verdict = f"{winner.upper()}, by {abs(diff):.3f}"
        note = "Small, but statistically significant."
    else:
        verdict = "A tie"
        note = "The difference is not statistically significant."
    c3.metric(f"Winner after {K} clicks", verdict, note, delta_color="off", delta_arrow="off",
              help="Paired bootstrap on the same test sessions, Holm-corrected for multiple comparisons.")

# --------------------------------------------------------------------------- all models
left, right = st.columns([1.2, 1], gap="large")
with left:
    st.subheader("All models, one scale", anchor=False)
    d = part.sort("pr_auc_mean")
    fig = go.Figure()
    for team in ("ml", "dl"):
        x = d.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]))
        fig.add_trace(go.Bar(
            y=[ui.label(m) for m in x["model"]], x=x["pr_auc_mean"], orientation="h",
            name=f"{ui.TEAMS[team]['title']} ({ui.TEAMS[team]['author']})",
            marker=dict(color=tcol[team], cornerradius=4),
            error_x=dict(type="data", array=x["pr_auc_std"].fill_null(0).to_list(), color=ink["muted"], thickness=1.5,
                         width=3),
            text=[f"{v:.4f}" for v in x["pr_auc_mean"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=ink["text"], size=12),
            customdata=x.select("pr_auc_std", "n_seeds").to_numpy(),
            hovertemplate="<b>%{y}</b><br>PR-AUC %{x:.4f} ± %{customdata[0]:.4f}"
                          "<br>%{customdata[1]} training runs (seeds)<extra></extra>",
        ))
    ui.style(fig, height=380, left=150, top=50, barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=[ui.label(m) for m in d["model"]], showgrid=False)
    fig.update_xaxes(range=[0, float(d["pr_auc_mean"].max()) * 1.35], title=f"PR-AUC after {K} clicks")
    ui.show(fig, key="all")
    st.caption("Mean of three training runs with different random seeds; the whisker is ± one standard deviation.")
with right:
    st.subheader("Across K (clicks seen)", anchor=False)
    rows = []
    for k_ in sorted(set(rq1["k"].to_list())):
        sub = rq1.filter(pl.col("k") == k_)
        a, b = best("ml", sub), best("dl", sub)
        if a and b:
            t = team_page.significant_vs(pairs, k_, a["model"], b["model"])
            w = "tie" if not (t and t[1]) else ("ML" if t[0] > 0 else "DL")
            rows.append({"K": k_, "Best ML": f"{ui.label(a['model'])} {a['pr_auc_mean']:.3f}",
                         "Best DL": f"{ui.label(b['model'])} {b['pr_auc_mean']:.3f}", "Winner": w})
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"K": st.column_config.NumberColumn(width=40),
                                "Winner": st.column_config.TextColumn(width=60)})
    if bm and bd:
        gap = abs(bm["pr_auc_mean"] - bd["pr_auc_mean"])
        lift = min(bm["pr_auc_lift_mean"], bd["pr_auc_lift_mean"])
        st.markdown(f"**In short:** at K={K} the two teams' best models are {gap:.4f} PR-AUC apart and both rank "
                    f"buyers at least {lift:.1f}× better than random. The choice between them comes down to speed "
                    "and simplicity.")

# --------------------------------------------------------------------------- head-to-head
@st.cache_data(show_spinner=False)
def seed_runs(path: str, mtime: float, mode: str, k: int) -> pl.DataFrame:
    """Test PR-AUC of every seed's latest rq1 run at K (older reruns of the same seed are dropped)."""
    del mtime                                   # cache key only: reload when all_runs.csv changes
    return (pl.read_csv(path, infer_schema_length=0)
            .filter((pl.col("experiment") == "rq1_models") & (pl.col("data_mode") == mode)
                    & (pl.col("k") == str(k)) & pl.col("test_pr_auc").is_not_null())
            .sort("started_at").unique(["model", "seed"], keep="last")
            .select("model", pl.col("seed").cast(pl.Int64), pl.col("test_pr_auc").cast(pl.Float64)))


top = learned.sort("pr_auc_mean", descending=True).head(2)
if top.height == 2:
    from src.config import resolve_path

    a, b = top["model"].to_list()
    st.divider()
    st.subheader(f"Head-to-head: {ui.label(a)} vs {ui.label(b)}", anchor=False)
    st.caption(f"The two most accurate models after {K} clicks. Each was trained three times with different random "
               "seeds and scored on the same test visits.")
    runs_csv = resolve_path(ui.cfg(), "all_runs_csv")
    seeds = (seed_runs(str(runs_csv), runs_csv.stat().st_mtime, ui.cfg()["data"]["mode"], K)
             if runs_csv.exists() else pl.DataFrame())
    row = (pairs.filter((pl.col("k") == K) & (((pl.col("model_a") == a) & (pl.col("model_b") == b))
                                             | ((pl.col("model_a") == b) & (pl.col("model_b") == a))))
           if pairs is not None else pl.DataFrame())
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        if seeds.height:
            wide = (seeds.filter(pl.col("model").is_in([a, b]))
                    .pivot(on="model", index="seed", values="test_pr_auc").sort("seed"))
            if a in wide.columns and b in wide.columns:
                wide = wide.with_columns(gap=pl.col(a) - pl.col(b))
                table = [{"Training run": f"seed {r['seed']}", ui.label(a): f"{r[a]:.4f}",
                          ui.label(b): f"{r[b]:.4f}", "Gap": f"{r['gap']:+.4f}",
                          "Ahead": ui.label(a) if r["gap"] > 0 else ui.label(b)} for r in wide.iter_rows(named=True)]
                ra, rb = top.row(0, named=True), top.row(1, named=True)
                table.append({"Training run": "mean ± sd", ui.label(a): f"{ra['pr_auc_mean']:.4f} ± {ra['pr_auc_std']:.4f}",
                              ui.label(b): f"{rb['pr_auc_mean']:.4f} ± {rb['pr_auc_std']:.4f}",
                              "Gap": f"{ra['pr_auc_mean'] - rb['pr_auc_mean']:+.4f}", "Ahead": ""})
                st.dataframe(table, hide_index=True, width="stretch")
        else:
            st.info("Per-seed results are not on this machine (the online demo keeps one seed per model).")
    with right:
        if row.height:
            r = row.row(0, named=True)
            sign = 1 if r["model_a"] == a else -1
            lo, hi = sorted((sign * r["diff_ci_low"], sign * r["diff_ci_high"]))
            sig = r["p_bootstrap_holm"] < team_page.ALPHA
            st.metric("Difference on the test visits", f"{sign * r['pr_auc_diff']:+.4f}",
                      f"95% CI {lo:+.4f} to {hi:+.4f}", delta_color="off", delta_arrow="off",
                      help="Paired bootstrap of the primary-seed models on the same test sessions.")
            st.metric("Verdict", "Significant" if sig else "A statistical tie",
                      f"p = {r['p_bootstrap_holm']:.2f} after Holm correction", delta_color="off", delta_arrow="off")
            st.markdown(
                (f"**{ui.label(a)} is genuinely ahead**, though by a small margin." if sig else
                 f"**They are equally accurate.** The gap is smaller than the noise from resampling the test visits"
                 f"{' and from retraining with another seed' if seeds.height else ''}, so rerunning either model "
                 "would not settle it. The engine therefore picks between them on speed (below)."))

# --------------------------------------------------------------------------- speed vs accuracy
st.divider()
left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("Accuracy vs speed", anchor=False)
    if eff is None:
        ui.missing("speed benchmarks", "05_benchmark_latency.py")
    else:
        e = ui.per_k(eff).filter(pl.col("model") != "rule_based")
        fig = go.Figure()
        by_x = e.sort("latency_b1_p95_ms")
        pos, prev = {}, None
        for m, y in zip(by_x["model"], by_x["test_pr_auc_primary"]):   # label above if higher than its left neighbour
            pos[m] = "top right" if prev is None or y >= prev else "bottom right"
            prev = y
        for team in ("ml", "dl"):
            x = e.filter(pl.col("model").is_in(ui.TEAMS[team]["models"]))
            fig.add_trace(go.Scatter(
                x=x["latency_b1_p95_ms"], y=x["test_pr_auc_primary"], mode="markers+text",
                name=ui.TEAMS[team]["title"], text=[ui.label(m) for m in x["model"]],
                textposition=[pos[m] for m in x["model"]],
                textfont=dict(color=ink["text"], size=11),
                marker=dict(size=13, color=tcol[team], line=dict(color=ink["surface"], width=2)),
                hovertemplate="<b>%{text}</b><br>%{x:.3f} ms · PR-AUC %{y:.4f}<extra></extra>",
            ))
        ui.style(fig, height=380, top=50)
        xs = e["latency_b1_p95_ms"]
        fig.update_xaxes(type="log", title="Time to score one session, p95 (ms, log scale)",
                         range=[float(np.log10(max(xs.min(), 1e-4) / 3)), float(np.log10(xs.max() * 12))])
        ys = e["test_pr_auc_primary"]
        fig.update_yaxes(title="PR-AUC", range=[float(ys.min()) - 0.01, float(ys.max()) + 0.012])
        ui.show(fig, key="speed")
        st.caption("Up and to the left is better. Everything is far below the 10 ms real-time budget.")
with right:
    st.subheader("What we would ship", anchor=False)
    if eff is not None and bm and bd:
        e = ui.per_k(eff)
        ship, tied = team_page.recommended(rq1, pairs, eff, K)
        other = [m for m in tied if ui.team_of(m) != ui.team_of(ship)]
        alt = min(other, key=lambda m: float(e.filter(pl.col("model") == m)["latency_b1_p95_ms"][0])) if other \
            else (bd if ui.team_of(ship) == "ml" else bm)["model"]

        def blurb(m: str) -> str:
            r, x = part.filter(pl.col("model") == m), e.filter(pl.col("model") == m)
            why = ("exact per-feature explanations (TreeSHAP)" if m in ("lightgbm", "random_forest")
                   else "explanations through SHAP")
            return (f"**{ui.label(m)}** ({ui.TEAMS[ui.team_of(m)]['title'].lower()}): PR-AUC {r['pr_auc_mean'][0]:.3f}, "
                    f"{x['latency_b1_p95_ms'][0]:.2f} ms per session, {x['model_size_mb'][0]:.1f} MB, {why}.")

        st.markdown(f"Statistically tied for the top at K={K}: {', '.join(ui.label(m) for m in tied)}. Among those "
                    "we pick the fastest.")
        st.markdown(f"**Default engine:** {blurb(ship)}\n\n**Alternative from the other team:** {blurb(alt)}")
        st.caption("Both are available in the live demo.")

# --------------------------------------------------------------------------- shared limit
if summary and summary.get("buyers_missed_by_all"):
    st.divider()
    st.subheader("A limit both teams hit", anchor=False)
    m = summary["buyers_missed_by_all"]
    c1, c2 = st.columns([1, 2.2])
    c1.metric("Buyers no model catches", f"{m['n'] / m['of']:.0%}", f"{m['n']:,} of {m['of']:,} test buyers",
              delta_color="off", delta_arrow="off")
    c2.markdown("These shoppers bought later, but their first clicks look exactly like browsing: none of them "
                "added anything to the cart. No model, linear, tree or neural, can tell them apart yet. "
                "That is a property of the data in the first few clicks, not a weakness of one method.")
