"""
app.py  –  LinkedIn Smart Interview Prep (AI-Powered Feature Prototype)
────────────────────────────────────────────────────────────────────────
Prototype of a new AI feature for LinkedIn's existing "Interview Prep" section.
The current LinkedIn feature shows generic questions unrelated to the job posting.
This prototype fixes that: paste any job description → get personalised questions,
answer them, and receive AI coaching feedback + a performance score.

Run locally:
    streamlit run app.py

Required env var:
    ANTHROPIC_API_KEY=your_key_here
"""

import streamlit as st
import time
import json
from nlp_pipeline import extract_keywords, score_answer, classify_question, model_is_trained, get_model_metrics
from ai_coach import generate_questions, get_feedback, get_session_summary

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  — LinkedIn blue palette
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn · Smart Interview Prep",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — LinkedIn-inspired design
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* LinkedIn-style white/light background */
    .stApp { background-color: #f3f2ef; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }

    /* LinkedIn blue header bar */
    .li-header {
        background: #0a66c2;
        border-radius: 10px;
        padding: 18px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .li-header h1 { color: white; margin: 0; font-size: 1.4rem; font-family: sans-serif; }
    .li-header p  { color: #cce0f5; margin: 0; font-size: 0.88rem; }

    /* Cards */
    .li-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px 24px;
        border: 1px solid #e0e0e0;
        margin-bottom: 14px;
    }

    /* Score card */
    .score-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .score-value { font-size: 2rem; font-weight: 700; }
    .score-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

    /* Question pill */
    .q-pill {
        background: #f8f9fa;
        border-left: 4px solid #0a66c2;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
        color: #1a1a1a;
        font-size: 0.97rem;
        line-height: 1.6;
    }
    .q-pill.technical  { border-left-color: #057642; }
    .q-pill.behavioral { border-left-color: #0a66c2; }

    /* Keyword chip */
    .kw-chip {
        display: inline-block;
        background: #e8f0fe;
        color: #0a66c2;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.78rem;
        margin: 3px;
        border: 1px solid #c5d9f7;
        font-weight: 500;
    }

    /* Existing feature badge */
    .existing-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .new-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 6px;
    }

    /* Feedback box */
    .feedback-box {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px 20px;
        color: #1a1a1a;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* Progress bar */
    .stProgress > div > div { background-color: #0a66c2 !important; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #0a66c2;
        border: none;
    }

    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 1,
        "job_desc": "",
        "job_title": "",
        "company": "",
        "keywords": [],
        "questions": {},
        "role_summary": "",
        "current_q_idx": 0,
        "all_questions": [],
        "session_log": [],
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def score_color(s):
    if s >= 75: return "#057642"
    if s >= 50: return "#d97706"
    return "#dc2626"

def render_score_cards(scores):
    mode = scores.get("mode", "heuristic")
    if mode == "ml_model":
        # ML model scores: ml_score, keyword_hit, overall + pred_label
        cols = st.columns(3)
        items = [
            ("ML Score",     scores.get("ml_score", 0)),
            ("Keyword Match",scores.get("keyword_hit", 0)),
            ("Overall",      scores.get("overall", 0)),
        ]
        for col, (label, val) in zip(cols, items):
            c = score_color(val)
            col.markdown(f"""
            <div class='score-card'>
                <div class='score-value' style='color:{c}'>{val}</div>
                <div class='score-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        pred  = scores.get("pred_label", "")
        conf  = scores.get("confidence", 0)
        emoji = {"weak": "🔴", "good": "🟡", "strong": "🟢"}.get(pred, "⚪")
        st.caption(f"Model prediction: {emoji} **{pred.capitalize()}** answer · confidence {conf}%  ·  *powered by trained RandomForest / LogisticRegression*")
    else:
        cols = st.columns(3)
        items = [
            ("Relevance",    scores.get("relevance", 0)),
            ("Completeness", scores.get("completeness", 0)),
            ("Keyword Match",scores.get("keyword_hit", 0)),
        ]
        for col, (label, val) in zip(cols, items):
            c = score_color(val)
            col.markdown(f"""
            <div class='score-card'>
                <div class='score-value' style='color:{c}'>{val}</div>
                <div class='score-label'>{label}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("*Scoring mode: heuristic fallback (run train_model.py to enable ML scoring)*")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # LinkedIn-style profile area
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:2.5rem">💼</div>
        <div style="font-weight:700; font-size:1rem; color:#1a1a1a">LinkedIn</div>
        <div style="font-size:0.8rem; color:#666">Smart Interview Prep</div>
        <div style="margin-top:6px">
            <span class="existing-badge">Existing Feature</span>
            <span class="new-badge">✨ AI Upgrade</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # What's new vs old callout
    with st.expander("💡 What's new vs existing?"):
        st.markdown("""
**Current LinkedIn Interview Prep:**
- ❌ Generic questions for any job
- ❌ No connection to the actual job posting
- ❌ No feedback on your answers
- ❌ No personalisation

**This AI upgrade adds:**
- ✅ Questions generated from the *real* job description
- ✅ NLP skill extraction (TF-IDF)
- ✅ ML-scored answers (0–100)
- ✅ Claude AI coaching feedback
- ✅ Behavioral + Technical split
        """)

    st.divider()

    api_key_input = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Needed for AI question generation and feedback"
    )
    if api_key_input:
        import os; os.environ["ANTHROPIC_API_KEY"] = api_key_input
        st.session_state.api_key_set = True
        st.success("Connected ✓", icon="✅")

    st.divider()

    difficulty = st.select_slider(
        "Difficulty",
        options=["Entry Level", "Mid Level", "Senior", "Lead / Principal"],
        value="Mid Level"
    )
    num_behavioral = st.number_input("Behavioural Qs", 1, 6, 3)
    num_technical  = st.number_input("Technical Qs",   1, 6, 3)

    st.divider()

    # ML model status
    if model_is_trained():
        m = get_model_metrics()
        st.success("🤖 ML Model: Active", icon="✅")
        st.caption(f"Best model: {m.get('best_model','')[:20]}")
        st.caption(f"Test accuracy: {m.get('test_accuracy',0)*100:.0f}%")
    else:
        st.warning("⚠️ ML model not trained. Run `python train_model.py` first.")

    if st.session_state.all_questions:
        total = len(st.session_state.all_questions)
        done  = len(st.session_state.session_log)
        st.markdown(f"**Progress: {done}/{total}**")
        st.progress(done / total if total else 0)

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_state()
        st.rerun()


# ─────────────────────────────────────────────────────────────
# STEP 1 — SETUP
# ─────────────────────────────────────────────────────────────
if st.session_state.step == 1:

    # LinkedIn-style header
    st.markdown("""
    <div class='li-header'>
        <div>
            <h1>💼 Smart Interview Prep</h1>
            <p>Paste a job description from LinkedIn and get a personalised mock interview — powered by AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Context: show the problem we're solving
    st.markdown("""
    <div class='li-card'>
        <span class="existing-badge">⚠️ Current LinkedIn Experience</span>
        <span style="color:#666; font-size:0.85rem; margin-left:10px">
            LinkedIn's existing Interview Prep shows the same generic questions to everyone, 
            regardless of the role you're applying for.
        </span>
        <br><br>
        <span class="new-badge">✨ This Prototype</span>
        <span style="color:#666; font-size:0.85rem; margin-left:10px">
            Generates questions directly from the job posting using NLP + AI, 
            scores your answers with a trained ML model, and provides coaching feedback.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### 📋 Job Details")

        col_a, col_b = st.columns(2)
        job_title = col_a.text_input("Job Title", placeholder="e.g. Senior Data Scientist")
        company   = col_b.text_input("Company",   placeholder="e.g. Google")

        job_desc = st.text_area(
            "Paste Job Description",
            placeholder="Copy & paste the full job description from the LinkedIn job posting here...\n\nE.g.: We are looking for a Senior Data Scientist to join our team. You will work with Python, SQL, TensorFlow...",
            height=260,
            label_visibility="collapsed"
        )

    with col_right:
        st.markdown("#### 🔍 How it works")
        st.markdown("""
        <div class='li-card' style='font-size:0.88rem; line-height:1.8'>
            <b>1. NLP Skill Extraction</b><br>
            TF-IDF algorithm scans the job description and identifies the most important skills and requirements<br><br>
            <b>2. AI Question Generation</b><br>
            Claude generates behavioural + technical questions tailored to <i>this specific role</i><br><br>
            <b>3. ML Answer Scoring</b><br>
            A trained classifier (Logistic Regression / Random Forest) predicts answer quality: <i>weak / good / strong</i><br><br>
            <b>4. AI Coaching Feedback</b><br>
            Personalised feedback on each answer with a model example
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_btn, _ = st.columns([1, 4])
    go = col_btn.button(
        "🚀 Start Interview Prep",
        type="primary",
        use_container_width=True,
        disabled=not job_desc.strip()
    )
    if not job_desc.strip():
        st.caption("⬆️ Paste a job description to get started")

    if go and job_desc.strip():
        with st.spinner("🔍 Analysing job description with NLP…"):
            keywords = extract_keywords(job_desc, top_n=12)
            st.session_state.keywords  = keywords
            st.session_state.job_desc  = job_desc
            st.session_state.job_title = job_title
            st.session_state.company   = company

        kw_html = "".join(f"<span class='kw-chip'>{kw}</span>" for kw in keywords)
        st.markdown(f"**Detected skills:** {kw_html}", unsafe_allow_html=True)
        time.sleep(0.4)

        if not st.session_state.api_key_set:
            st.warning("No API key — using demo questions. Add your key in the sidebar for personalised questions.")
            questions = {
                "role_summary": f"{job_title or 'This role'} at {company or 'the company'}",
                "behavioral": [
                    "Tell me about a time you had to learn a new skill quickly to deliver a project.",
                    "Describe a situation where you disagreed with a team decision. How did you handle it?",
                    f"Give an example of how you have used {keywords[0] if keywords else 'a key skill'} to solve a real business problem.",
                ],
                "technical": [
                    f"Explain your experience with {keywords[0] if keywords else 'the main technology in this role'}.",
                    f"How would you approach designing a scalable solution involving {keywords[1] if len(keywords) > 1 else 'the core technology'}?",
                    "Walk me through how you ensure quality and reliability in your work.",
                ]
            }
        else:
            with st.spinner("🤖 Generating personalised questions with Claude AI…"):
                questions = generate_questions(
                    job_desc, keywords,
                    num_behavioral=int(num_behavioral),
                    num_technical=int(num_technical),
                    difficulty=difficulty,
                )

        st.session_state.questions    = questions
        st.session_state.role_summary = questions.get("role_summary", "")

        all_q = []
        for q in questions.get("behavioral", []):
            all_q.append({"question": q, "type": "Behavioral"})
        for q in questions.get("technical", []):
            all_q.append({"question": q, "type": "Technical"})

        st.session_state.all_questions = all_q
        st.session_state.step = 2
        st.rerun()


# ─────────────────────────────────────────────────────────────
# STEP 2 — PRACTICE
# ─────────────────────────────────────────────────────────────
elif st.session_state.step == 2:

    all_q = st.session_state.all_questions
    idx   = st.session_state.current_q_idx
    log   = st.session_state.session_log

    if idx >= len(all_q):
        st.session_state.step = 3
        st.rerun()

    current = all_q[idx]
    q_text  = current["question"]
    q_type  = current["type"]
    total   = len(all_q)

    # Header
    title_str = st.session_state.job_title or "Your Role"
    company_str = f" at {st.session_state.company}" if st.session_state.company else ""
    st.markdown(f"""
    <div class='li-header'>
        <div>
            <h1>💼 Mock Interview · {title_str}{company_str}</h1>
            <p>Answer each question as you would in a real interview. AI will score and coach you.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(idx / total, text=f"Question {idx+1} of {total}")
    st.divider()

    col_q, col_hist = st.columns([3, 1], gap="large")

    with col_q:
        # Question type badge
        badge_color = "#0a66c2" if q_type == "Behavioral" else "#057642"
        badge_bg    = "#e8f0fe" if q_type == "Behavioral" else "#d1fae5"
        st.markdown(
            f'<span style="background:{badge_bg}; color:{badge_color}; '
            f'border:1px solid {badge_color}44; border-radius:999px; '
            f'padding:4px 14px; font-size:0.8rem; font-weight:600">'
            f'{"🤝 Behavioural" if q_type == "Behavioral" else "⚙️ Technical"}</span>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="q-pill {q_type.lower()}" style="margin-top:10px">❓ {q_text}</div>',
                    unsafe_allow_html=True)

        if q_type == "Behavioral":
            with st.expander("💡 STAR Method tip"):
                st.markdown("""
| | |
|---|---|
| **S**ituation | Set the context — where and when |
| **T**ask | What was YOUR responsibility |
| **A**ction | Specific steps you personally took |
| **R**esult | Measurable outcome / impact |
                """)

        st.markdown("**✍️ Your Answer**")
        answer = st.text_area(
            "answer",
            placeholder="Type your answer here... Aim for 80–150 words for a complete response.",
            height=175,
            key=f"ans_{idx}",
            label_visibility="collapsed"
        )

        if answer:
            wc = len(answer.split())
            wc_col = "#057642" if 80 <= wc <= 200 else "#d97706" if wc > 30 else "#dc2626"
            st.markdown(
                f'<span style="color:{wc_col}; font-size:0.8rem">'
                f'Word count: {wc} {"✓" if 80 <= wc <= 200 else "— aim for 80–150"}</span>',
                unsafe_allow_html=True
            )

        bc1, bc2, _ = st.columns([1.5, 1.5, 3])
        submit = bc1.button("✅ Submit", type="primary", use_container_width=True,
                            disabled=not answer.strip())
        skip   = bc2.button("⏭️ Skip",  use_container_width=True)

        if submit and answer.strip():
            with st.spinner("Scoring your answer…"):
                scores = score_answer(q_text, answer, st.session_state.job_desc, q_type)

            st.markdown("#### 📊 Your Score")
            render_score_cards(scores)

            with st.spinner("Getting AI coaching feedback…"):
                if st.session_state.api_key_set:
                    feedback = get_feedback(q_text, answer, q_type, scores, st.session_state.job_desc)
                else:
                    feedback = (
                        "**What worked well ✅**\n"
                        "- You gave a direct response to the question\n\n"
                        "**Areas to improve 🔧**\n"
                        "- Add specific examples with measurable outcomes\n"
                        "- Use the STAR structure for behavioural questions\n\n"
                        "**💡 Add your Anthropic API key in the sidebar for personalised AI feedback.**"
                    )

            st.markdown("#### 🤖 AI Coach Feedback")
            st.markdown(f'<div class="feedback-box">{feedback}</div>', unsafe_allow_html=True)

            st.session_state.session_log.append({
                "question": q_text, "type": q_type,
                "answer": answer, "scores": scores, "feedback": feedback,
            })

            st.divider()
            label = "➡️ Next Question" if idx + 1 < total else "📋 View Report"
            if st.button(label, type="primary"):
                st.session_state.current_q_idx += 1
                st.rerun()

        if skip:
            st.session_state.session_log.append({
                "question": q_text, "type": q_type, "answer": "[Skipped]",
                "scores": {"overall": 0}, "feedback": "Skipped.",
            })
            st.session_state.current_q_idx += 1
            st.rerun()

    with col_hist:
        st.markdown("#### 📝 Progress")
        if log:
            for i, entry in enumerate(log):
                ov = entry["scores"].get("overall", 0)
                c  = score_color(ov)
                pred = entry["scores"].get("pred_label", "")
                emoji = {"weak":"🔴","good":"🟡","strong":"🟢"}.get(pred, "")
                st.markdown(
                    f'<div style="background:#fff;border-radius:8px;padding:8px 10px;'
                    f'margin-bottom:6px;border:1px solid #e0e0e0;border-left:3px solid {c}">'
                    f'<span style="font-size:0.72rem;color:#666">Q{i+1} · {entry["type"]}</span><br>'
                    f'<span style="color:{c};font-weight:700">{ov}/100</span> {emoji}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("Answered questions appear here.")

        st.markdown("#### 🔑 Key Skills")
        for kw in st.session_state.keywords[:8]:
            st.markdown(f"<span class='kw-chip'>{kw}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STEP 3 — REPORT
# ─────────────────────────────────────────────────────────────
elif st.session_state.step == 3:

    log      = st.session_state.session_log
    answered = [e for e in log if e["answer"] != "[Skipped]"]

    title_str   = st.session_state.job_title or "Your Role"
    company_str = f" at {st.session_state.company}" if st.session_state.company else ""

    st.markdown(f"""
    <div class='li-header'>
        <div>
            <h1>📋 Interview Report · {title_str}{company_str}</h1>
            <p>Here's how you performed across all questions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not answered:
        st.warning("No questions answered. Click Start Over.")
        st.stop()

    avg_overall = round(sum(e["scores"]["overall"] for e in answered) / len(answered))

    if avg_overall >= 80:   grade, gc = "Excellent 🌟", "#057642"
    elif avg_overall >= 65: grade, gc = "Strong 💪",    "#065f46"
    elif avg_overall >= 50: grade, gc = "Good 👍",      "#d97706"
    else:                   grade, gc = "Needs Work 📚", "#dc2626"

    # Aggregate scores depending on mode
    mode = answered[0]["scores"].get("mode", "heuristic")
    mc = st.columns(4)
    if mode == "ml_model":
        agg_items = [
            ("Overall",      round(sum(e["scores"]["overall"]   for e in answered)/len(answered))),
            ("ML Score",     round(sum(e["scores"].get("ml_score",0) for e in answered)/len(answered))),
            ("Keyword Match",round(sum(e["scores"].get("keyword_hit",0) for e in answered)/len(answered))),
            ("Questions",    len(answered)),
        ]
    else:
        agg_items = [
            ("Overall",      avg_overall),
            ("Relevance",    round(sum(e["scores"].get("relevance",0)    for e in answered)/len(answered))),
            ("Completeness", round(sum(e["scores"].get("completeness",0) for e in answered)/len(answered))),
            ("Questions",    len(answered)),
        ]

    for col, (label, val) in zip(mc, agg_items):
        c = score_color(val) if label != "Questions" else "#0a66c2"
        col.markdown(f"""
        <div class='score-card'>
            <div class='score-value' style='color:{c}'>{val}</div>
            <div class='score-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<p style="text-align:center;font-size:1.3rem;font-weight:700;color:{gc};margin:12px 0">'
                f'Performance: {grade}</p>', unsafe_allow_html=True)

    st.divider()

    col_summary, col_detail = st.columns([2, 3], gap="large")

    with col_summary:
        st.markdown("### 🤖 AI Coach Summary")
        with st.spinner("Generating summary…"):
            summary = get_session_summary(answered) if st.session_state.api_key_set else (
                "**Session complete!** Add your Anthropic API key to receive a personalised "
                "summary with strengths, improvement areas, and actionable next steps."
            )
        st.markdown(f'<div class="feedback-box">{summary}</div>', unsafe_allow_html=True)

        st.markdown("#### Score Breakdown")
        import pandas as pd
        if mode == "ml_model":
            chart_df = pd.DataFrame({
                "Score": [
                    round(sum(e["scores"].get("ml_score",0)   for e in answered)/len(answered)),
                    round(sum(e["scores"].get("keyword_hit",0) for e in answered)/len(answered)),
                ]
            }, index=["ML Quality Score", "Keyword Match"])
        else:
            chart_df = pd.DataFrame({
                "Score": [
                    round(sum(e["scores"].get("relevance",0)    for e in answered)/len(answered)),
                    round(sum(e["scores"].get("completeness",0) for e in answered)/len(answered)),
                    round(sum(e["scores"].get("keyword_hit",0)  for e in answered)/len(answered)),
                ]
            }, index=["Relevance", "Completeness", "Keyword Match"])
        st.bar_chart(chart_df, color="#0a66c2")

        # Label distribution if ML model was used
        if mode == "ml_model":
            labels = [e["scores"].get("pred_label","") for e in answered]
            strong = labels.count("strong")
            good   = labels.count("good")
            weak   = labels.count("weak")
            st.markdown(f"**Answer quality breakdown:** 🟢 {strong} Strong · 🟡 {good} Good · 🔴 {weak} Weak")

    with col_detail:
        st.markdown("### 📝 Question Review")
        tab_all, tab_beh, tab_tech = st.tabs(["All Questions", "Behavioural", "Technical"])

        def render_entries(entries):
            for i, e in enumerate(entries):
                ov   = e["scores"].get("overall", 0)
                pred = e["scores"].get("pred_label","")
                emoji= {"weak":"🔴","good":"🟡","strong":"🟢"}.get(pred, "")
                with st.expander(f"Q{i+1} · {e['type']} · {ov}/100 {emoji} — {e['question'][:55]}…"):
                    st.markdown(f"**Question:** {e['question']}")
                    st.markdown(f"**Your Answer:** {e['answer']}")
                    render_score_cards(e["scores"])
                    st.markdown("**AI Feedback:**")
                    st.markdown(f'<div class="feedback-box">{e["feedback"]}</div>',
                                unsafe_allow_html=True)

        with tab_all:  render_entries(answered)
        with tab_beh:  render_entries([e for e in answered if e["type"]=="Behavioral"])
        with tab_tech: render_entries([e for e in answered if e["type"]=="Technical"])

    st.divider()

    report = json.dumps({
        "job_title":     st.session_state.job_title,
        "company":       st.session_state.company,
        "role_summary":  st.session_state.role_summary,
        "keywords":      st.session_state.keywords,
        "grade":         grade,
        "avg_overall":   avg_overall,
        "questions":     answered,
    }, indent=2)

    c1, c2, _ = st.columns([2, 2, 4])
    c1.download_button("⬇️ Download Report", data=report,
                       file_name="linkedin_interview_report.json",
                       mime="application/json", use_container_width=True)
    if c2.button("🔄 New Interview", type="primary", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_state()
        st.rerun()
