"""
LinkedIn Smart Interview Prep — v2
"""

import streamlit as st
import json
import re
from nlp_pipeline import extract_keywords, score_answer
from ai_coach import generate_questions, analyse_answer
import plotly.graph_objects as go
from rag_engine import RAGEngine
from interview_kb import get_knowledge_base
from chatbot import chat, get_initial_message
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Smart Interview Prep",
    page_icon="in",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise RAG engine once ────────────────────────────────────────────────
@st.cache_resource
def load_rag_engine():
    kb = get_knowledge_base()
    return RAGEngine(kb)

rag_engine = load_rag_engine()

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .navbar {
    background: #0a66c2; padding: 12px 24px; border-radius: 8px;
    display: flex; align-items: center; gap: 12px; margin-bottom: 24px;
  }
  .navbar-logo { color: white; font-size: 22px; font-weight: 700; }
  .navbar-badge {
    background: rgba(255,255,255,0.2); color: white;
    padding: 3px 10px; border-radius: 12px; font-size: 12px;
  }
  .progress-bar-container {
    display: flex; justify-content: space-between;
    margin-bottom: 28px; position: relative;
  }
  .step { text-align: center; flex: 1; }
  .step-circle {
    width: 32px; height: 32px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 14px; margin-bottom: 6px;
  }
  .step-active .step-circle { background: #0a66c2; color: white; }
  .step-done .step-circle { background: #057642; color: white; }
  .step-inactive .step-circle { background: #e0e0e0; color: #666; }
  .step-label { font-size: 12px; color: #555; }
  .question-card {
    border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
    background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .pill {
    display: inline-block; padding: 3px 10px;
    border-radius: 12px; font-size: 11px; font-weight: 600;
    margin-right: 6px;
  }
  .pill-behavioural-active { background: #e8f4fd; color: #0a66c2; }
  .pill-technical-active { background: #fef3e2; color: #b45309; }
  .pill-inactive { background: #f0f0f0; color: #aaa; }
  .score-card {
    border-radius: 10px; padding: 16px 20px; margin-top: 12px;
  }
  .score-strong { background: #ecfdf5; border-left: 4px solid #057642; }
  .score-good   { background: #fffbeb; border-left: 4px solid #d97706; }
  .score-weak   { background: #fef2f2; border-left: 4px solid #dc2626; }
  .rag-context-box {
    background: #f0f7ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
    font-size: 12px; color: #1e40af;
  }
  .chat-user { background: #0a66c2; color: white; border-radius: 16px 16px 4px 16px; padding: 10px 14px; margin: 6px 0; max-width: 80%; margin-left: auto; }
  .chat-assistant { background: #f3f4f6; color: #111; border-radius: 16px 16px 16px 4px; padding: 10px 14px; margin: 6px 0; max-width: 80%; }
  .section-divider { border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }
  .keyword-chip {
    display: inline-block; background: #e8f4fd; color: #0a66c2;
    border-radius: 12px; padding: 3px 10px; font-size: 12px; margin: 3px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "step": 1,
    "job_title": "",
    "company": "",
    "job_description": "",
    "difficulty": "Mid-level",
    "keywords": [],
    "questions": [],
    "current_q": 0,
    "answers": [],
    "overall_score": 0,
    "submitted": {},
    "chat_history": [],
    "chat_initialised": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <span class="navbar-logo">in</span>
  <span class="navbar-logo" style="font-size:16px;">LinkedIn</span>
  <span class="navbar-badge">Smart Interview Prep AI</span>
</div>
""", unsafe_allow_html=True)

# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current_step):
    steps = ["Setup", "Practice", "Report", "Coach"]
    html = '<div class="progress-bar-container">'
    for i, label in enumerate(steps, 1):
        if i < current_step:
            cls = "step-done"
        elif i == current_step:
            cls = "step-active"
        else:
            cls = "step-inactive"
        html += f'<div class="step {cls}"><div class="step-circle">{i}</div><div class="step-label">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

progress_bar(st.session_state["step"])

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — Setup
# ════════════════════════════════════════════════════════════════════════
if st.session_state["step"] == 1:
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("Set up your interview")
        st.session_state["job_title"] = st.text_input("Job Title", st.session_state["job_title"], placeholder="e.g. Senior Product Manager")
        st.session_state["company"] = st.text_input("Company", st.session_state["company"], placeholder="e.g. Spotify")
        st.session_state["difficulty"] = st.select_slider(
            "Experience level",
            options=["Entry Level", "Mid-level", "Senior", "Lead / Principal"],
            value=st.session_state["difficulty"],
        )
        st.session_state["job_description"] = st.text_area(
            "Paste the job description",
            st.session_state["job_description"],
            height=220,
            placeholder="Paste the full job description here...",
        )

        if st.button("Generate Interview", type="primary", use_container_width=True):
            if not st.session_state["job_title"] or not st.session_state["job_description"]:
                st.warning("Please fill in the job title and job description.")
            else:
                with st.spinner("Analysing job description and generating questions..."):
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
                            {"question": f"How would you approach building a roadmap for {st.session_state['job_title']}?", "type": "technical", "skill": "product strategy"},
                            {"question": "Tell me about a time you handled a disagreement with a stakeholder.", "type": "behavioural", "skill": "conflict resolution"},
                            {"question": "Walk me through how you would prioritise a backlog with 80 items.", "type": "technical", "skill": "prioritisation"},
                        ]
                    st.session_state["questions"] = qs
                    st.session_state["answers"] = []
                    st.session_state["current_q"] = 0
                    st.session_state["submitted"] = {}
                    st.session_state["step"] = 2
                    st.rerun()

    with col_side:
        st.markdown("#### How it works")
        st.markdown("""
        1. Paste a real job description
        2. Answer tailored questions
        3. Get AI-scored feedback
        4. Chat with your personal coach
        """)
        st.info("New in v2: Feedback is grounded in a knowledge base of real example answers using RAG.")

# ════════════════════════════════════════════════════════════════════════
# STEP 2 — Practice
# ════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 2:
    questions = st.session_state["questions"]
    total = len(questions)
    answered = len(st.session_state["answers"])
    idx = st.session_state["current_q"]

    col_main, col_side = st.columns([2, 1])

    with col_side:
        st.markdown("#### Your progress")
        st.progress(answered / total)
        st.caption(f"{answered} of {total} answered")
        st.markdown("#### Role requirements")
        kw_html = "".join(f'<span class="keyword-chip">{k}</span>' for k in st.session_state["keywords"][:10])
        st.markdown(kw_html, unsafe_allow_html=True)

    with col_main:
        if idx < total:
            q = questions[idx]
            q_type = q.get("type", "behavioural")

            b_cls = "pill-behavioural-active" if q_type == "behavioural" else "pill-inactive"
            t_cls = "pill-technical-active" if q_type == "technical" else "pill-inactive"

            st.markdown(f"""
            <div class="question-card">
              <div style="margin-bottom:10px;">
                <span class="pill {b_cls}">Behavioural</span>
                <span class="pill {t_cls}">Technical</span>
              </div>
              <p style="font-size:17px; font-weight:600; margin:0;">{q['question']}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("STAR method tip"):
                st.markdown("""
                **S**ituation — Set the scene briefly  
                **T**ask — What was your responsibility?  
                **A**ction — What did YOU specifically do?  
                **R**esult — What was the measurable outcome?
                """)

            answer_key = f"answer_{idx}"
            answer = st.text_area(
                "Your answer",
                key=answer_key,
                height=160,
                placeholder="Write your answer here...",
            )
            word_count = len(answer.split()) if answer.strip() else 0
            st.caption(f"{word_count} words")

            col_sub, col_skip = st.columns([1, 1])

            with col_sub:
                submitted_key = f"submitted_{idx}"
                if not st.session_state["submitted"].get(submitted_key):
                    if st.button("Submit Answer", type="primary", use_container_width=True):
                        if not answer.strip():
                            st.warning("Please write an answer before submitting.")
                        else:
                            with st.spinner("Analysing answer (step 1: competencies, step 2: feedback)..."):
                                result_score = score_answer(
                                    question=q["question"],
                                    answer=answer,
                                    job_description=st.session_state["job_description"],
                                    question_type=q_type,
                                )
                                ml_score = result_score.get("ml_score", result_score.get("relevance", 50))
                                kw_score = result_score.get("keyword_hit", 50)
                                overall = result_score.get("overall", 50)

                                # RAG: retrieve similar examples
                                rag_context = rag_engine.build_rag_context(query=answer, k=2)

                                # Two-call pipeline: Call A (competencies) → Call B (feedback)
                                result = analyse_answer(
                                    question=q["question"],
                                    answer=answer,
                                    question_type=q_type,
                                    job_title=st.session_state["job_title"],
                                    keywords=st.session_state["keywords"],
                                    rag_context=rag_context,
                                )

                                entry = {
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
                                }
                                st.session_state["answers"].append(entry)
                                st.session_state["submitted"][submitted_key] = True
                                st.rerun()
                else:
                    # Show result
                    entry = next(
                        (a for a in st.session_state["answers"] if a["question"] == q["question"]),
                        None,
                    )
                    if entry:
                        score = entry["score"]
                        if score >= 75:
                            grade, cls = "Strong", "score-strong"
                        elif score >= 50:
                            grade, cls = "Good", "score-good"
                        else:
                            grade, cls = "Needs work", "score-weak"

                        st.markdown(f"""
                        <div class="score-card {cls}">
                          <strong>{score}/100 — {grade}</strong><br>
                          <small>ML Quality: {entry['ml_score']}/100 &nbsp;|&nbsp; Keyword Match: {entry['kw_score']}/100</small><br><br>
                          {entry['feedback']}
                        </div>
                        """, unsafe_allow_html=True)

                        if entry.get("strengths"):
                            st.markdown("**Strengths:** " + " · ".join(entry["strengths"]))
                        if entry.get("improvements"):
                            st.markdown("**To improve:** " + " · ".join(entry["improvements"]))

                        # Show which KB examples were used (transparency)
                        with st.expander("View retrieved examples used for feedback"):
                            st.markdown(
                                f'<div class="rag-context-box">{entry["rag_context_used"].replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True,
                            )

                        # Next question
                        if st.button("Next Question", type="primary", use_container_width=True):
                            st.session_state["current_q"] += 1
                            if st.session_state["current_q"] >= total:
                                scores = [a["score"] for a in st.session_state["answers"]]
                                st.session_state["overall_score"] = round(sum(scores) / len(scores)) if scores else 0
                                st.session_state["step"] = 3
                            st.rerun()

            with col_skip:
                if not st.session_state["submitted"].get(f"submitted_{idx}"):
                    if st.button("Skip", use_container_width=True):
                        st.session_state["answers"].append({
                            "question": q["question"],
                            "type": q_type,
                            "answer": "[skipped]",
                            "score": 0,
                            "ml_score": 0,
                            "kw_score": 0,
                            "feedback": "Question skipped.",
                            "strengths": [],
                            "improvements": [],
                            "rag_context_used": "",
                        })
                        st.session_state["submitted"][f"submitted_{idx}"] = True
                        st.session_state["current_q"] += 1
                        if st.session_state["current_q"] >= total:
                            scores = [a["score"] for a in st.session_state["answers"] if a["answer"] != "[skipped]"]
                            st.session_state["overall_score"] = round(sum(scores) / len(scores)) if scores else 0
                            st.session_state["step"] = 3
                        st.rerun()

# ════════════════════════════════════════════════════════════════════════
# STEP 3 — Report
# ════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 3:
    overall = st.session_state["overall_score"]
    answers = st.session_state["answers"]

    if overall >= 75:
        grade_label, grade_color = "Strong Candidate", "#057642"
    elif overall >= 50:
        grade_label, grade_color = "Good Candidate", "#d97706"
    else:
        grade_label, grade_color = "Needs Preparation", "#dc2626"

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Score", f"{overall}/100")
    col2.metric("Questions", len(answers))
    col3.metric("Assessment", grade_label)

    st.markdown(f"<hr class='section-divider'>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Competency Radar", "Score per Question", "Download Report"])

    with tab1:
        # ── Python post-processing: aggregate competency scores across all answers ──
        comp_keys = ["structure", "specificity", "impact", "relevance", "communication"]
        agg = {k: [] for k in comp_keys}
        for a in answers:
            for k in comp_keys:
                val = a.get("competencies", {}).get(k)
                if val is not None:
                    agg[k].append(val)
        avg_comps = {k: round(sum(v) / len(v)) if v else 0 for k, v in agg.items()}

        # Radar chart
        labels = [c.capitalize() for c in comp_keys]
        values = [avg_comps[k] for k in comp_keys]
        values_closed = values + [values[0]]
        labels_closed = labels + [labels[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(10, 102, 194, 0.15)",
            line=dict(color="#0a66c2", width=2),
            name="Your profile",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(t=40, b=40),
            height=380,
            title="Average competency profile across all answers",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        weakest = min(avg_comps, key=avg_comps.get)
        strongest = max(avg_comps, key=avg_comps.get)
        st.markdown(f"**Strongest dimension:** {strongest.capitalize()} ({avg_comps[strongest]}/100)  \n"
                    f"**Most room to improve:** {weakest.capitalize()} ({avg_comps[weakest]}/100)")

    with tab2:
        # Bar chart — score per question
        q_labels = [f"Q{i+1}" for i in range(len(answers))]
        q_scores = [a["score"] for a in answers]
        colors = ["#057642" if s >= 75 else "#d97706" if s >= 50 else "#dc2626" for s in q_scores]

        fig_bar = go.Figure(go.Bar(
            x=q_labels,
            y=q_scores,
            marker_color=colors,
            text=q_scores,
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis=dict(range=[0, 110], title="Score /100"),
            xaxis_title="Question",
            title="Score per question",
            height=340,
            margin=dict(t=40, b=40),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        for i, a in enumerate(answers, 1):
            score = a["score"]
            color = "#057642" if score >= 75 else "#d97706" if score >= 50 else "#dc2626"
            with st.expander(f"Q{i}: {a['question'][:70]}... — {score}/100"):
                st.markdown(f"**Your answer:** {a['answer']}")
                st.markdown(f"**Score:** <span style='color:{color};font-weight:600'>{score}/100</span>", unsafe_allow_html=True)
                st.markdown(f"**Feedback:** {a['feedback']}")

    with tab3:
        report = {
            "job_title": st.session_state["job_title"],
            "company": st.session_state["company"],
            "overall_score": overall,
            "answers": answers,
        }
        st.download_button(
            "Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="interview_report.json",
            mime="application/json",
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    if st.button("Chat with your AI Coach", type="primary", use_container_width=True):
        st.session_state["step"] = 4
        st.rerun()

    if st.button("Start New Interview", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

# ════════════════════════════════════════════════════════════════════════
# STEP 4 — AI Coach Chatbot
# ════════════════════════════════════════════════════════════════════════
elif st.session_state["step"] == 4:
    st.subheader("AI Interview Coach")
    st.caption("Ask me about your answers, how to improve, or request example answers.")

    session_data = {
        "job_title": st.session_state["job_title"],
        "company": st.session_state["company"],
        "keywords": st.session_state["keywords"],
        "overall_score": st.session_state["overall_score"],
        "answers": st.session_state["answers"],
    }

    # Initialise with opening message
    if not st.session_state["chat_initialised"]:
        opening = get_initial_message(session_data)
        st.session_state["chat_history"] = [{"role": "assistant", "content": opening}]
        st.session_state["chat_initialised"] = True

    # Render chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask your coach...")
    if user_input:
        with st.spinner("Coaching..."):
            response = chat(
                user_message=user_input,
                conversation_history=st.session_state["chat_history"],
                session_data=session_data,
            )
        st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    if st.button("Back to Report"):
        st.session_state["step"] = 3
        st.rerun()
