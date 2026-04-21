import streamlit as st
import json
from nlp_pipeline import extract_keywords, score_answer
from ai_coach import generate_questions, analyse_answer, build_candidate_mirror, build_prep_plan
import plotly.graph_objects as go
from rag_engine import RAGEngine
from interview_kb import get_knowledge_base
from chatbot import chat, opening_message
from refine import refine_answer   
import os

st.set_page_config(
    page_title="LinkedIn · Smart Interview Prep",
    page_icon="in",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_rag_engine():
    # V3: default to semantic embeddings; falls back to TF-IDF if unavailable
    return RAGEngine(get_knowledge_base(), mode="semantic")

rag_engine = load_rag_engine()

# ── V3: dev mode routing ──────────────────────────────────────────────────
# If URL has ?dev=1, render the evaluation UI instead of the candidate app.
# This keeps V2 (the candidate experience) unchanged and adds V3 alongside.
if st.query_params.get("dev") == "1":
    from dev_mode import render_dev_mode
    render_dev_mode()
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f3f2ef; }

.top-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0 18px 0; margin-bottom: 4px;
}
.top-bar-logo { font-size: 28px; font-weight: 800; color: #0a66c2; }
.top-bar-title { font-size: 15px; font-weight: 600; color: #191919; }
.top-bar-badge {
    background: #0a66c2; color: white; font-size: 11px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px;
}

.step-tabs {
    display: flex; border-radius: 8px; overflow: hidden;
    border: 1px solid #ddd; margin-bottom: 28px; background: white;
}
.step-tab {
    flex: 1; padding: 11px 0; text-align: center;
    font-size: 13px; font-weight: 500; color: #666;
    background: white; border: none;
}
.step-tab-active {
    background: #0a66c2; color: white; font-weight: 600;
}
.step-tab-done { background: #f0f7ff; color: #0a66c2; }

.question-card {
    background: white; border-radius: 8px;
    border: 1px solid #e0e0e0; padding: 20px 24px;
    margin-bottom: 14px;
}
.q-meta { font-size: 12px; color: #666; margin-bottom: 10px; }
.q-text { font-size: 16px; font-weight: 600; color: #191919; line-height: 1.5; }

.pill {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin-right: 6px;
}
.pill-b-on  { background: #e8f4fd; color: #0a66c2; }
.pill-t-on  { background: #fff3e0; color: #e65100; }
.pill-off   { background: #f0f0f0; color: #aaa; }

.side-card {
    background: white; border-radius: 8px;
    border: 1px solid #e0e0e0; padding: 16px 18px; margin-bottom: 14px;
}
.side-title { font-size: 13px; font-weight: 700; color: #191919; margin-bottom: 10px; }
.kw-chip {
    display: inline-block; background: #f0f7ff; color: #0a66c2;
    border-radius: 20px; padding: 3px 10px; font-size: 12px; margin: 3px 2px;
}

.score-box {
    border-radius: 8px; padding: 14px 18px; margin-top: 10px;
}
.score-strong { background: #f0fdf4; border-left: 3px solid #16a34a; }
.score-good   { background: #fffbeb; border-left: 3px solid #d97706; }
.score-weak   { background: #fef2f2; border-left: 3px solid #dc2626; }

.chat-user {
    background: #0a66c2; color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; margin: 6px 0;
    max-width: 78%; margin-left: auto; font-size: 14px;
}
.chat-bot {
    background: white; color: #191919; border: 1px solid #e0e0e0;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px; margin: 6px 0;
    max-width: 78%; font-size: 14px;
}
.new-feature-banner {
    background: #f0f7ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #1e40af; margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
defaults = {
    "step": 1,
    "job_title": "", "company": "", "job_description": "",
    "difficulty": "Mid-level", "keywords": [],
    "questions": [], "current_q": 0, "answers": [],
    "overall_score": 0, "submitted": {},
    "chat_history": [], "chat_ready": False,
    "mirror": None, "prep_plan": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <span class="top-bar-logo">in</span>
  <span class="top-bar-title">Smart Interview Prep</span>
  <span class="top-bar-badge">AI</span>
</div>
""", unsafe_allow_html=True)

# ── step tabs ─────────────────────────────────────────────────────────────────
def step_tabs(current):
    labels = ["1 · Setup", "2 · Practice", "3 · Report", "4 · Coach"]
    parts = []
    for i, label in enumerate(labels, 1):
        if i == current:
            cls = "step-tab step-tab-active"
        elif i < current:
            cls = "step-tab step-tab-done"
        else:
            cls = "step-tab"
        parts.append(f'<div class="{cls}">{label}</div>')
    st.markdown(f'<div class="step-tabs">{"".join(parts)}</div>', unsafe_allow_html=True)

step_tabs(st.session_state["step"])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Setup
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["step"] == 1:
    st.markdown('<div class="new-feature-banner">New: feedback now references real example answers from a curated knowledge base, and you can chat with an AI coach after your session.</div>', unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("#### Prepare for your interview")
        st.caption("Paste the job description from LinkedIn — we'll generate questions tailored to this specific role.")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state["job_title"] = st.text_input("Job Title", st.session_state["job_title"], placeholder="e.g. Senior Data Scientist")
        with c2:
            st.session_state["company"] = st.text_input("Company", st.session_state["company"], placeholder="e.g. Google")

        st.session_state["difficulty"] = st.select_slider(
            "Experience level",
            options=["Entry Level", "Mid-level", "Senior", "Lead / Principal"],
            value=st.session_state["difficulty"],
        )
        st.session_state["job_description"] = st.text_area(
            "Job Description",
            st.session_state["job_description"],
            height=200,
            placeholder="Paste the full job description here...",
        )

        col_btn, col_hint = st.columns([1, 2])
        with col_btn:
            generate = st.button("Generate Questions →", type="primary", use_container_width=True)
        with col_hint:
            st.caption("Paste a job description to get started")

        if generate:
            if not st.session_state["job_title"] or not st.session_state["job_description"]:
                st.warning("Please fill in the job title and job description.")
            else:
                with st.spinner("Analysing job description..."):
                    kw = extract_keywords(st.session_state["job_description"])
                    st.session_state["keywords"] = kw
                    try:
                        qs = generate_questions(
                            st.session_state["job_title"],
                            st.session_state["company"],
                            st.session_state["job_description"],
                            st.session_state["difficulty"],
                        )
                    except Exception:
                        qs = [
                            {"question": "Tell me about a time you led a cross-functional project.", "type": "behavioural", "skill": "leadership"},
                            {"question": "Describe a situation where you had to make a data-driven decision.", "type": "behavioural", "skill": "analytical thinking"},
                            {"question": f"How would you build a roadmap for {st.session_state['job_title']}?", "type": "technical", "skill": "strategy"},
                            {"question": "Tell me about a time you handled a disagreement with a stakeholder.", "type": "behavioural", "skill": "conflict resolution"},
                            {"question": "How do you prioritise a backlog with competing requests?", "type": "technical", "skill": "prioritisation"},
                        ]
                    st.session_state["questions"] = qs
                    st.session_state["answers"] = []
                    st.session_state["current_q"] = 0
                    st.session_state["submitted"] = {}
                    st.session_state["step"] = 2
                    st.rerun()

    with col_side:
        st.markdown('<div class="side-card"><div class="side-title">How it works</div>', unsafe_allow_html=True)
        st.markdown("""
1. Paste a real job description  
2. Answer tailored questions  
3. Get AI-scored feedback with real examples  
4. Chat with your personal coach
""")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Practice
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 2:
    questions = st.session_state["questions"]
    total = len(questions)
    idx = st.session_state["current_q"]
    answered = len(st.session_state["answers"])

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.markdown(f'<div class="side-card"><div class="side-title">{st.session_state["job_title"]} at {st.session_state["company"]}</div>', unsafe_allow_html=True)
        st.caption("Key requirements")
        kw_html = "".join(f'<span class="kw-chip">{k}</span>' for k in st.session_state["keywords"][:8])
        st.markdown(kw_html + "</div>", unsafe_allow_html=True)

        if answered > 0:
            st.markdown('<div class="side-card">', unsafe_allow_html=True)
            st.progress(answered / total)
            st.caption(f"{answered} of {total} answered")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        if idx < total:
            q = questions[idx]
            q_type = q.get("type", "behavioural")

            b_cls = "pill-b-on" if q_type == "behavioural" else "pill-off"
            t_cls = "pill-t-on" if q_type == "technical" else "pill-off"

            st.markdown(f"""
            <div class="q-meta">Question {idx+1} of {total} · {st.session_state['job_title']} at {st.session_state['company']}</div>
            <div style="margin-bottom:10px;">
              <span class="pill {b_cls}">Behavioural</span>
              <span class="pill {t_cls}">Technical</span>
            </div>
            <div class="question-card">
              <div class="q-text">{q['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("STAR Method tip"):
                st.markdown("""
**S**ituation — Set the scene briefly  
**T**ask — What was your responsibility?  
**A**ction — What did YOU specifically do?  
**R**esult — What was the measurable outcome?
""")

            submitted_key = f"submitted_{idx}"
            if not st.session_state["submitted"].get(submitted_key):
                answer = st.text_area(
                    "Your answer",
                    key=f"answer_{idx}",
                    height=160,
                    placeholder="Type your answer here... Aim for 80–150 words.",
                    label_visibility="collapsed",
                )
                st.caption(f"{len(answer.split()) if answer.strip() else 0} words")

                col_sub, col_skip, _ = st.columns([1, 1, 2])
                with col_sub:
                    if st.button("Submit →", type="primary", use_container_width=True):
                        if not answer.strip():
                            st.warning("Please write an answer first.")
                        else:
                            with st.spinner("Scoring your answer..."):
                                result_score = score_answer(
                                    question=q["question"],
                                    answer=answer,
                                    job_description=st.session_state["job_description"],
                                    question_type=q_type,
                                )
                                ml_score = result_score.get("ml_score", result_score.get("relevance", 50))
                                kw_score = result_score.get("keyword_hit", 50)
                                overall = result_score.get("overall", 50)

                                rag_context = rag_engine.build_rag_context(query=answer, k=2)

                                result = analyse_answer(
                                    question=q["question"],
                                    answer=answer,
                                    question_type=q_type,
                                    job_title=st.session_state["job_title"],
                                    keywords=st.session_state["keywords"],
                                    rag_context=rag_context,
                                )

                                st.session_state["answers"].append({
                                    "question": q["question"],
                                    "type": q_type,
                                    "answer": answer,
                                    "score": overall,
                                    "ml_score": ml_score,
                                    "kw_score": kw_score,
                                    "feedback": result.get("feedback", ""),
                                    "strengths": result.get("strengths", []),
                                    "improvements": result.get("improvements", []),
                                    "competencies": result.get("competencies", {}),
                                    "rag_context_used": rag_context,
                                })
                                st.session_state["submitted"][submitted_key] = True
                                st.rerun()
                with col_skip:
                    if st.button("Skip question", use_container_width=True):
                        st.session_state["answers"].append({
                            "question": q["question"], "type": q_type,
                            "answer": "[skipped]", "score": 0,
                            "ml_score": 0, "kw_score": 0,
                            "feedback": "Skipped.", "strengths": [],
                            "improvements": [], "competencies": {}, "rag_context_used": "",
                        })
                        st.session_state["submitted"][submitted_key] = True
                        st.session_state["current_q"] += 1
                        if st.session_state["current_q"] >= total:
                            scores = [a["score"] for a in st.session_state["answers"] if a["answer"] != "[skipped]"]
                            st.session_state["overall_score"] = round(sum(scores)/len(scores)) if scores else 0
                            st.session_state["step"] = 3
                        st.rerun()
            else:
                entry = next((a for a in st.session_state["answers"] if a["question"] == q["question"]), None)
                if entry:
                    score = entry["score"]
                    if score >= 75:
                        grade, cls = "Strong", "score-strong"
                    elif score >= 50:
                        grade, cls = "Good", "score-good"
                    else:
                        grade, cls = "Needs work", "score-weak"

                    st.markdown(f"""
                    <div class="score-box {cls}">
                      <strong>{score}/100 — {grade}</strong><br>
                      <small style="color:#666">ML quality: {entry['ml_score']}/100 &nbsp;·&nbsp; Keyword match: {entry['kw_score']}/100</small><br><br>
                      {entry['feedback']}
                    </div>
                    """, unsafe_allow_html=True)

                    if entry.get("strengths"):
                        st.markdown("**Strengths:** " + "  ·  ".join(entry["strengths"]))
                    if entry.get("improvements"):
                        st.markdown("**To improve:** " + "  ·  ".join(entry["improvements"]))

                    with st.expander("View examples used for feedback"):
                        st.caption(entry["rag_context_used"])

                    if st.button("Next question →", type="primary"):
                        st.session_state["current_q"] += 1
                        if st.session_state["current_q"] >= total:
                            scores = [a["score"] for a in st.session_state["answers"] if a["answer"] != "[skipped]"]
                            st.session_state["overall_score"] = round(sum(scores)/len(scores)) if scores else 0
                            st.session_state["step"] = 3
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Report
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 3:
    overall = st.session_state["overall_score"]
    answers = st.session_state["answers"]

    if overall >= 75:
        grade = "Strong Candidate"
    elif overall >= 50:
        grade = "Good Candidate"
    else:
        grade = "Needs Preparation"

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Score", f"{overall}/100")
    c2.metric("Questions answered", len([a for a in answers if a["answer"] != "[skipped]"]))
    c3.metric("Assessment", grade)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Competency Profile", "Score per Question", "Candidate Mirror", "7-Day Prep Plan", "Download"])

    with tab1:
        dims = ["structure", "specificity", "impact", "relevance", "communication"]
        agg = {d: [] for d in dims}
        for a in answers:
            for d in dims:
                v = a.get("competencies", {}).get(d)
                if v is not None:
                    agg[d].append(v)
        avg = {d: round(sum(v)/len(v)) if v else 0 for d, v in agg.items()}

        vals = [avg[d] for d in dims]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[d.capitalize() for d in dims] + [dims[0].capitalize()],
            fill="toself",
            fillcolor="rgba(10,102,194,0.12)",
            line=dict(color="#0a66c2", width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=360,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        weakest = min(avg, key=avg.get)
        strongest = max(avg, key=avg.get)
        st.caption(f"Strongest: {strongest.capitalize()} ({avg[strongest]}/100)  ·  Most room to improve: {weakest.capitalize()} ({avg[weakest]}/100)")

    with tab2:
        q_labels = [f"Q{i+1}" for i in range(len(answers))]
        q_scores = [a["score"] for a in answers]
        colors = ["#16a34a" if s >= 75 else "#d97706" if s >= 50 else "#dc2626" for s in q_scores]
        fig2 = go.Figure(go.Bar(x=q_labels, y=q_scores, marker_color=colors, text=q_scores, textposition="outside"))
        fig2.update_layout(yaxis=dict(range=[0, 115], title="Score /100"), height=320, margin=dict(t=20, b=20), plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

        for i, a in enumerate(answers, 1):
            # Show the most recent version's score as the headline, but keep
            # the full revision history accessible below.
            latest = a["revisions"][-1] if a.get("revisions") else a
            score = latest["score"]
            color = "#16a34a" if score >= 75 else "#d97706" if score >= 50 else "#dc2626"
            rev_badge = f" · Refined ×{len(a['revisions'])}" if a.get("revisions") else ""

            with st.expander(f"Q{i}: {a['question'][:70]}...{rev_badge}"):
                # ── Original answer ─────────────────────────────────────────
                st.markdown("**Original answer:**")
                st.markdown(f"_{a['answer']}_")
                st.markdown(
                    f"**Original score:** <span style='color:{('#16a34a' if a['score']>=75 else '#d97706' if a['score']>=50 else '#dc2626')};font-weight:600'>{a['score']}/100</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Original feedback:** {a['feedback']}")

                # ── Revision history (each revision side-by-side with the prior) ──
                if a.get("revisions"):
                    for r_idx, rev in enumerate(a["revisions"], 1):
                        st.markdown("---")
                        st.markdown(f"**Revision {r_idx}**")
                        rc = "#16a34a" if rev["score"] >= 75 else "#d97706" if rev["score"] >= 50 else "#dc2626"
                        delta = rev.get("overall_delta", 0)
                        delta_color = "#16a34a" if delta > 0 else "#dc2626" if delta < 0 else "#666"
                        delta_sign = f"{delta:+d}" if delta != 0 else "±0"
                        st.markdown(f"_{rev['answer']}_")
                        st.markdown(
                            f"**Revised score:** <span style='color:{rc};font-weight:600'>{rev['score']}/100</span>"
                            f" &nbsp; <span style='color:{delta_color};font-weight:600'>({delta_sign})</span>",
                            unsafe_allow_html=True,
                        )

                        # Per-dimension delta chips
                        deltas = rev.get("deltas", {})
                        if deltas:
                            chips = []
                            for dim, d_val in deltas.items():
                                if d_val > 0:
                                    chips.append(f"<span style='background:#f0fdf4;color:#16a34a;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;'>{dim} {d_val:+d}</span>")
                                elif d_val < 0:
                                    chips.append(f"<span style='background:#fef2f2;color:#dc2626;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;'>{dim} {d_val:+d}</span>")
                                else:
                                    chips.append(f"<span style='background:#f1f5f9;color:#666;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;'>{dim} ±0</span>")
                            st.markdown("".join(chips), unsafe_allow_html=True)

                        if rev.get("summary_of_change"):
                            st.markdown(
                                f"<div style='background:#f8fafc;border-left:3px solid #0a66c2;padding:10px 12px;margin-top:8px;font-size:13px;'><strong>What changed:</strong> {rev['summary_of_change']}</div>",
                                unsafe_allow_html=True,
                            )

                # ── Refine this answer ──────────────────────────────────────
                st.markdown("---")
                refine_key = f"refine_open_{i}"
                text_key = f"refine_text_{i}"
                if refine_key not in st.session_state:
                    st.session_state[refine_key] = False

                if not st.session_state[refine_key]:
                    if st.button(f"✏️ Refine this answer", key=f"btn_open_{i}"):
                        st.session_state[refine_key] = True
                        st.rerun()
                else:
                    st.markdown("**Write a revised version of your answer:**")
                    st.caption("Try addressing the specific feedback above. Add concrete numbers, use STAR structure, or cut the vague parts.")
                    default_text = latest["answer"]
                    revised = st.text_area(
                        "Revised answer",
                        value=default_text,
                        height=160,
                        key=text_key,
                        label_visibility="collapsed",
                    )
                    col_submit, col_cancel = st.columns([1, 1])
                    with col_submit:
                        if st.button("Submit revision", key=f"btn_submit_{i}", type="primary", use_container_width=True):
                            if revised.strip() and revised.strip() != latest["answer"].strip():
                                with st.spinner("Re-scoring your revised answer..."):
                                    try:
                                        rev = refine_answer(
                                            answer_idx=i - 1,
                                            revised_text=revised,
                                            job_title=st.session_state["job_title"],
                                            keywords=st.session_state["keywords"],
                                            rag_engine=rag_engine,
                                        )
                                        # Append to revisions list on the original answer
                                        if "revisions" not in st.session_state["answers"][i - 1]:
                                            st.session_state["answers"][i - 1]["revisions"] = []
                                        st.session_state["answers"][i - 1]["revisions"].append(rev)

                                        # Recompute overall score using the latest version of each answer
                                        latest_scores = []
                                        for a_obj in st.session_state["answers"]:
                                            if a_obj.get("answer") == "[skipped]":
                                                continue
                                            last = a_obj["revisions"][-1] if a_obj.get("revisions") else a_obj
                                            latest_scores.append(last["score"])
                                        if latest_scores:
                                            st.session_state["overall_score"] = round(sum(latest_scores) / len(latest_scores))

                                        st.session_state[refine_key] = False
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Could not re-score the revision: {e}")
                            else:
                                st.warning("Please write a different version of the answer.")
                    with col_cancel:
                        if st.button("Cancel", key=f"btn_cancel_{i}", use_container_width=True):
                            st.session_state[refine_key] = False
                            st.rerun()


    with tab3:
        st.caption("How did you come across? The AI reads all your answers together and builds a picture of you as a candidate — the same way an interviewer would.")
        if st.session_state["mirror"] is None:
            if st.button("Generate Candidate Mirror", type="primary"):
                with st.spinner("Analysing your answers as a whole..."):
                    try:
                        mirror = build_candidate_mirror(
                            answers=answers,
                            job_title=st.session_state["job_title"],
                            keywords=st.session_state["keywords"],
                        )
                        st.session_state["mirror"] = mirror
                        st.rerun()
                    except Exception:
                        st.error("Could not generate mirror. Try again.")
        else:
            mirror = st.session_state["mirror"]
            st.markdown("#### How you come across")
            st.markdown(f'<div class="question-card"><div class="q-text" style="font-size:15px;font-weight:400;font-style:italic;">"{mirror.get("how_you_come_across", "")}"</div></div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Communication style**")
                st.markdown(mirror.get("communication_style", ""))
                st.markdown("**Hidden strength**")
                st.markdown(f'<div style="background:#f0fdf4;border-radius:8px;padding:12px 14px;border-left:3px solid #16a34a;font-size:14px;">{mirror.get("hidden_strength", "")}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown("**Blind spots**")
                for b in mirror.get("blind_spots", []):
                    st.markdown(f"- {b}")
                st.markdown("**Doubts the interviewer is probably left with**")
                for d in mirror.get("interviewer_doubts", []):
                    st.markdown(f'<div style="background:#fef2f2;border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:13px;">❓ {d}</div>', unsafe_allow_html=True)

    with tab4:
        st.caption("A personalised 7-day plan based on your weakest competency dimensions and the role requirements.")
        if st.session_state["prep_plan"] is None:
            if st.button("Generate My Prep Plan", type="primary"):
                with st.spinner("Building your personalised plan..."):
                    try:
                        dims = ["structure", "specificity", "impact", "relevance", "communication"]
                        agg = {d: [] for d in dims}
                        for a in answers:
                            for d in dims:
                                v = a.get("competencies", {}).get(d)
                                if v is not None:
                                    agg[d].append(v)
                        avg_scores = {d: round(sum(v)/len(v)) if v else 50 for d, v in agg.items()}
                        plan = build_prep_plan(
                            answers=answers,
                            job_title=st.session_state["job_title"],
                            keywords=st.session_state["keywords"],
                            competency_averages=avg_scores,
                        )
                        st.session_state["prep_plan"] = plan
                        st.rerun()
                    except Exception:
                        st.error("Could not generate plan. Try again.")
        else:
            plan = st.session_state["prep_plan"]
            focus = plan.get("focus_areas", [])
            if focus:
                st.markdown("**Focus areas:** " + "  ·  ".join(focus))
            st.markdown("")
            for day in plan.get("days", []):
                with st.expander(f"Day {day['day']} — {day['title']}"):
                    st.markdown(day["task"])
            if plan.get("key_reminder"):
                st.markdown(f'<div style="background:#fffbeb;border-radius:8px;padding:14px 16px;border-left:3px solid #d97706;margin-top:16px;"><strong>Morning of the interview:</strong><br>{plan["key_reminder"]}</div>', unsafe_allow_html=True)

    with tab5:
        report = {
            "job_title": st.session_state["job_title"],
            "company": st.session_state["company"],
            "overall_score": overall,
            "answers": answers,
            "mirror": st.session_state.get("mirror"),
            "prep_plan": st.session_state.get("prep_plan"),
        }
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="interview_report.json",
            mime="application/json",
        )

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Chat with your AI Coach →", type="primary", use_container_width=True):
            st.session_state["step"] = 4
            st.rerun()
    with col_b:
        if st.button("Start new interview", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Coach Chatbot
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 4:
    col_main, col_side = st.columns([2, 1])

    with col_side:
        st.markdown(f'<div class="side-card"><div class="side-title">{st.session_state["job_title"]} at {st.session_state["company"]}</div>', unsafe_allow_html=True)
        st.caption(f"Overall score: {st.session_state['overall_score']}/100")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("← Back to report"):
            st.session_state["step"] = 3
            st.rerun()

    with col_main:
        st.markdown("#### AI Interview Coach")
        st.caption("Ask me anything about your answers — I'll help you improve.")

        session_data = {
            "job_title": st.session_state["job_title"],
            "company": st.session_state["company"],
            "keywords": st.session_state["keywords"],
            "overall_score": st.session_state["overall_score"],
            "answers": st.session_state["answers"],
        }

        if not st.session_state["chat_ready"]:
            msg = opening_message(session_data)
            st.session_state["chat_history"] = [{"role": "assistant", "content": msg}]
            st.session_state["chat_ready"] = True

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask your coach...")
        if user_input:
            with st.spinner("Thinking..."):
                chat(
                    user_message=user_input,
                    history=st.session_state["chat_history"],
                    session_data=session_data,
                )
            st.rerun()
