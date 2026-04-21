"""
dev_mode.py
───────────
The V3 evaluation UI. Accessed via ?dev=1 URL parameter.

Separated from the candidate-facing app because:
  - Evaluation tools have a different audience (me, the developer) than the
    candidate app (interview-prep users)
  - Keeps app.py clean and unchanged in structure
  - Mirrors how real products separate end-user UI from internal admin panels

Two tabs:
  1. Eval Lab  — run a test case through all 4 variants, save the result
  2. Analytics — aggregate findings across all saved evaluation runs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from eval_testset import get_test_set
from eval_harness import run_evaluation
from user_votes import vote_summary, load_all_votes
from eval_storage import save_run, load_all_runs, delete_run


COMPETENCY_DIMS = ["structure", "specificity", "impact", "relevance", "communication"]


# ─────────────────────────────────────────────────────────────────────────────
# Dev-mode header (different from the candidate-facing top bar)
# ─────────────────────────────────────────────────────────────────────────────
def _dev_header():
    st.markdown("""
    <div style="background:#1e293b; color:#f1f5f9; padding:12px 18px;
                border-radius:8px; margin-bottom:18px; display:flex;
                justify-content:space-between; align-items:center;">
      <div>
        <div style="font-size:13px; font-weight:700; letter-spacing:2px;
                    color:#94a3b8; text-transform:uppercase;">
          V3 · Evaluation Mode
        </div>
        <div style="font-size:16px; font-weight:600; margin-top:2px;">
          Smart Interview Prep — Internal Evaluation Layer
        </div>
      </div>
      <div style="font-size:12px; color:#94a3b8;">
        Candidate view: remove <code style="color:#fbbf24;">?dev=1</code> from URL
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EVAL LAB
# ─────────────────────────────────────────────────────────────────────────────
def _render_retrieval_table(retrievals, title):
    st.markdown(f"**{title}**")
    if not retrievals:
        st.caption("No retrievals.")
        return
    rows = []
    for i, r in enumerate(retrievals, 1):
        rows.append({
            "Rank": i,
            "Skill": r.get("skill", ""),
            "Similarity": f"{r.get('similarity', 0):.3f}",
            "Preview": r.get("text_preview", "")[:80],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_scores_table(scores_by_variant):
    """scores_by_variant: dict of {variant_key: {dim: score}}"""
    if not scores_by_variant:
        return
    rows = []
    for dim in COMPETENCY_DIMS:
        row = {"Dimension": dim.capitalize()}
        for variant_key, scores in scores_by_variant.items():
            row[variant_key] = scores.get(dim, "—")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_eval_lab():
    st.markdown("#### Eval Lab")
    st.caption("Run a single test case through all 4 variants (TF-IDF × V1-single, TF-IDF × V2-two-step, Semantic × V1-single, Semantic × V2-two-step) and compare outputs side-by-side.")

    test_cases = get_test_set()

    # Init session state for this tab
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None
    if "eval_running" not in st.session_state:
        st.session_state["eval_running"] = False

    # Select a test case
    col_pick, col_run = st.columns([3, 1])
    with col_pick:
        case_labels = [f"{c['id']}  —  {c['expected_quality_tier']}  ·  {c['expected_skill']}"
                       for c in test_cases]
        selected_idx = st.selectbox(
            "Select test case",
            options=list(range(len(test_cases))),
            format_func=lambda i: case_labels[i],
            key="eval_case_select",
        )
        selected_case = test_cases[selected_idx]

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Evaluation", type="primary", use_container_width=True,
                            disabled=st.session_state["eval_running"])

    # Show the selected test case context
    with st.expander("Test case details", expanded=False):
        st.markdown(f"**Question:** {selected_case['question']}")
        st.markdown(f"**Answer:** _{selected_case['answer']}_")
        st.markdown(f"**Job:** {selected_case['job_title']}  ·  **Type:** {selected_case['question_type']}")
        if selected_case.get("notes"):
            st.info(f"**Notes:** {selected_case['notes']}")

    # Run the eval
    if run_btn:
        st.session_state["eval_running"] = True
        with st.spinner("Running 4 variants... this takes ~30 seconds (2 LLM calls per v2 variant, 1 per v1 variant)"):
            try:
                result = run_evaluation(selected_case)
                st.session_state["eval_result"] = result
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.session_state["eval_result"] = None
        st.session_state["eval_running"] = False
        st.rerun()

    # Show result if we have one
    result = st.session_state["eval_result"]
    if result is None:
        st.info("No evaluation run yet. Pick a test case above and click **Run Evaluation**.")
        return

    st.markdown("---")
    st.markdown(f"#### Result  ·  `{result['test_case']['id']}`  ·  run_id `{result['run_id']}`")

    variants = result["variants"]

    # ── Retrieval comparison ──
    st.markdown("##### 1. Retrieval comparison")
    st.caption("Same answer → two retrieval methods → how much do the top-3 results overlap?")

    cols = st.columns(2)
    with cols[0]:
        _render_retrieval_table(
            variants["tfidf__v2_two_step"]["top_3_retrievals"],
            "TF-IDF top 3"
        )
    with cols[1]:
        _render_retrieval_table(
            variants["semantic__v2_two_step"]["top_3_retrievals"],
            "Semantic top 3"
        )

    overlap = result["comparisons"].get("retrieval_overlap_v2", {})
    if overlap:
        st.markdown(
            f"**Top-3 overlap:** {overlap.get('overlap_count', 0)}/3 documents match  ·  "
            f"Jaccard similarity: {overlap.get('jaccard', 0):.2f}"
        )

    # ── Score comparison ──
    st.markdown("")
    st.markdown("##### 2. Score comparison (V1 single-prompt vs V2 two-step)")
    st.caption("Same answer + same retrieval (semantic) → do V1 and V2 feedback pipelines produce the same scores?")

    score_table = {
        "V1-single-prompt (semantic)": variants["semantic__v1_single"]["scores"],
        "V2-two-step (semantic)":      variants["semantic__v2_two_step"]["scores"],
    }
    _render_scores_table(score_table)

    div = result["comparisons"].get("score_divergence_v1_vs_v2_semantic", {})
    if div:
        st.markdown(
            f"**Mean absolute difference across dimensions:** {div.get('mean_abs_diff', 0):.1f} points"
        )

    # ── Feedback comparison ──
    st.markdown("")
    st.markdown("##### 3. Feedback text comparison")
    fb_cols = st.columns(2)
    with fb_cols[0]:
        st.markdown("**V1 (single prompt)**")
        st.markdown(f"_{variants['semantic__v1_single']['feedback']}_")
    with fb_cols[1]:
        st.markdown("**V2 (two-step)**")
        st.markdown(f"_{variants['semantic__v2_two_step']['feedback']}_")

    # ── Save / discard ──
    st.markdown("---")
    col_save, col_discard = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save this run to evaluations/", type="primary", use_container_width=True):
            path = save_run(result)
            st.success(f"Saved to `{path.name}`")
            st.session_state["eval_result"] = None
            st.rerun()
    with col_discard:
        if st.button("🗑️ Discard", use_container_width=True):
            st.session_state["eval_result"] = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def render_analytics():
    st.markdown("#### Analytics across all saved runs")
    st.caption("Aggregates findings across every evaluation run in `evaluations/runs/`.")

    runs = load_all_runs()
    if not runs:
        st.info("No saved runs yet. Go to the **Eval Lab** tab, run some test cases, and save them here.")
        return

    st.markdown(f"**{len(runs)} run(s) loaded from disk.**")
   
    # ── V3: user vote aggregates ──────────────────────────────────────────
    votes = load_all_votes()
    if votes:
        summary = vote_summary()
        st.markdown("---")
        st.markdown("##### User votes on feedback fairness")
        st.caption(
            "Votes collected during normal candidate use. 'Does this feedback feel "
            "fair to you?' — with an optional comment on the 'not really' path."
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total votes", summary["total"])
        col_b.metric("Felt fair", f"{summary['up']}  ({summary['up_pct']}%)")
        col_c.metric("Felt off", summary["down"])

        # Show the down-vote comments — these are gold for the 2-pager
        down_with_comments = [v for v in votes if not v.get("helpful") and v.get("comment", "").strip()]
        if down_with_comments:
            st.markdown("**What candidates said felt off:**")
            for v in down_with_comments[-5:]:   # show most recent 5
                st.markdown(
                    f'<div style="padding:8px 12px;background:#fffbeb;border-left:3px solid #d97706;'
                    f'border-radius:4px;margin-bottom:6px;font-size:13px;color:#854d0e;">'
                    f'<em>"{v["comment"]}"</em> '
                    f'<span style="color:#999;font-size:11px;">— on a {v.get("score","?")}/100 answer</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            f"With N={summary['total']}, these numbers are directional. They address the "
            f"professor's V2 suggestion to let users vote, without pretending N=4 is a study."
        )
        st.markdown("---")

    # ── Run-by-run summary table ──
    st.markdown("##### Per-run summary")
    rows = []
    for r in runs:
        tc = r.get("test_case", {})
        comp = r.get("comparisons", {})
        overlap = comp.get("retrieval_overlap_v2", {})
        divergence = comp.get("score_divergence_v1_vs_v2_semantic", {})
        rows.append({
            "Run": r.get("_filename", "")[:40],
            "Test case": tc.get("id", ""),
            "Expected tier": tc.get("expected_quality_tier", ""),
            "Retrieval overlap (top-3)": f"{overlap.get('overlap_count', 0)}/3",
            "V1 vs V2 score diff (mean abs)": f"{divergence.get('mean_abs_diff', 0):.1f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Chart 1: retrieval overlap per test case ──
    st.markdown("##### Retrieval overlap: does semantic retrieve different documents than TF-IDF?")
    st.caption("Top-3 overlap. 3/3 = identical retrievals (no difference). 0/3 = fully different. Lower = semantic and TF-IDF disagreed more.")

    chart_rows = []
    for r in runs:
        overlap = r.get("comparisons", {}).get("retrieval_overlap_v2", {})
        chart_rows.append({
            "case": r.get("test_case", {}).get("id", ""),
            "overlap": overlap.get("overlap_count", 0),
        })
    if chart_rows:
        cdf = pd.DataFrame(chart_rows)
        fig = go.Figure(go.Bar(
            x=cdf["case"], y=cdf["overlap"],
            marker_color="#0a66c2",
            text=cdf["overlap"], textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 3.5], title="Docs in common (top 3)"),
            xaxis=dict(title=""),
            height=340, margin=dict(t=30, b=20),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_overlap = cdf["overlap"].mean()
        differs = (cdf["overlap"] < 3).sum()
        st.markdown(
            f"**Across {len(cdf)} runs:** avg overlap = {avg_overlap:.2f}/3. "
            f"Semantic produced different top-3 retrievals than TF-IDF in "
            f"**{differs}/{len(cdf)}** cases."
        )

    # ── Chart 2: score divergence V1 vs V2 per test case ──
    st.markdown("##### Score divergence: does V2 two-step give different scores than V1 single-prompt?")
    st.caption("Mean absolute difference across 5 dimensions, holding retrieval (semantic) constant. Higher = V1 and V2 disagreed more. This is the 'scoring drift' effect V2 was designed to fix.")

    div_rows = []
    for r in runs:
        d = r.get("comparisons", {}).get("score_divergence_v1_vs_v2_semantic", {})
        div_rows.append({
            "case": r.get("test_case", {}).get("id", ""),
            "mean_abs_diff": d.get("mean_abs_diff", 0),
        })
    if div_rows:
        ddf = pd.DataFrame(div_rows)
        fig2 = go.Figure(go.Bar(
            x=ddf["case"], y=ddf["mean_abs_diff"],
            marker_color="#dc2626",
            text=ddf["mean_abs_diff"].round(1), textposition="outside",
        ))
        fig2.update_layout(
            yaxis=dict(title="Mean absolute score diff (0-100 scale)"),
            xaxis=dict(title=""),
            height=340, margin=dict(t=30, b=20),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            f"**Across {len(ddf)} runs:** average score divergence between "
            f"V1 single-prompt and V2 two-step = **{ddf['mean_abs_diff'].mean():.1f} points**. "
        )

    # ── Raw run browser ──
    st.markdown("---")
    st.markdown("##### Inspect individual runs")
    for r in runs:
        with st.expander(f"📄 {r.get('_filename', '')}", expanded=False):
            tc = r.get("test_case", {})
            st.markdown(f"**Test case:** `{tc.get('id', '')}`")
            st.markdown(f"**Question:** {tc.get('question', '')}")
            st.markdown(f"**Answer:** _{tc.get('answer', '')}_")
            st.markdown(f"**Timestamp:** {r.get('timestamp_utc', '')}")
            st.json(r.get("variants", {}), expanded=False)
            if st.button(f"Delete this run", key=f"del_{r.get('_filename', '')}"):
                delete_run(r.get("_filename", ""))
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — called by app.py when ?dev=1
# ─────────────────────────────────────────────────────────────────────────────
def render_dev_mode():
    _dev_header()
    tab1, tab2 = st.tabs(["🧪 Eval Lab", "📊 Analytics"])
    with tab1:
        render_eval_lab()
    with tab2:
        render_analytics()
