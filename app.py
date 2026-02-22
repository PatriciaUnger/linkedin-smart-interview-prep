"""
app.py - LinkedIn Smart Interview Prep
Clean, LinkedIn-style UI
"""
import streamlit as st
import json
from nlp_pipeline import extract_keywords, score_answer, classify_question, model_is_trained, get_model_metrics
from ai_coach import generate_questions, get_feedback, get_session_summary

st.set_page_config(
    page_title="LinkedIn · Smart Interview Prep",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

* { font-family: 'Source Sans Pro', sans-serif; box-sizing: border-box; }

.stApp { background: #f3f2ef; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* LinkedIn Navbar */
.li-nav {
    background: #ffffff;
    border-bottom: 1px solid #e0e0e0;
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    height: 52px;
    position: sticky;
    top: 0;
    z-index: 100;
    margin-bottom: 0;
}
.li-nav-logo {
    font-size: 1.6rem;
    font-weight: 900;
    color: #0a66c2;
    letter-spacing: -1px;
}
.li-nav-feature {
    font-size: 0.8rem;
    color: #666;
    border-left: 1px solid #e0e0e0;
    padding-left: 16px;
    margin-left: 4px;
}

/* Main card */
.li-card {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    padding: 24px;
    margin-bottom: 12px;
}

/* Job input area */
.li-section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 16px;
}

/* Question card */
.q-card {
    background: #fff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    padding: 28px 32px;
    margin-bottom: 16px;
}
.q-type-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.q-type-behavioral {
    background: #e8f0fe;
    color: #0a66c2;
}
.q-type-technical {
    background: #e6f4ea;
    color: #057642;
}
.q-text {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a1a1a;
    line-height: 1.5;
    margin-bottom: 20px;
}

/* Skill chips */
.skill-chip {
    display: inline-block;
    background: #eef3fb;
    color: #0a66c2;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}

/* Score display */
.score-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.score-box {
    flex: 1;
    background: #f8f9fa;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.score-num {
    font-size: 1.8rem;
    font-weight: 700;
}
.score-lbl {
    font-size: 0.72rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 2px;
}

/* Progress steps */
.steps {
    display: flex;
    gap: 0;
    margin-bottom: 24px;
    background: #fff;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    overflow: hidden;
}
.step {
    flex: 1;
    padding: 12px;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 600;
    color: #999;
    border-right: 1px solid #e0e0e0;
}
.step:last-child { border-right: none; }
.step.active { background: #0a66c2; color: white; }
.step.done { background: #e6f4ea; color: #057642; }

/* Feedback box */
.feedback-box {
    background: #f0f7ff;
    border-left: 3px solid #0a66c2;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #1a1a1a;
    margin-top: 16px;
}

/* Primary button override */
.stButton > button[kind="primary"] {
    background: #0a66c2 !important;
    color: white !important;
    border: none !important;
    border-radius: 24px !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #004182 !important;
}
.stButton > button {
    border-radius: 24px !important;
    font-weight: 600 !important;
}

/* Text areas and inputs */
.stTextArea textarea {
    border-radius: 8px !important;
    border-color: #e0e0e0 !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #0a66c2 !important;
    box-shadow: 0 0 0 2px #0a66c233 !important;
}
.stTextInput input {
    border-radius: 8px !important;
}

/* Word count */
.word-count {
    font-size: 0.78rem;
    color: #999;
    text-align: right;
    margin-top: 4px;
}
.word-count.good { color: #057642; }
.word-count.warn { color: #d97706; }
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ──────────────────────────────────────────────────
st.markdown("""
<div style="background:#ffffff;border-bottom:1px solid #e0e0e0;padding:12px 24px;
            display:flex;align-items:center;gap:16px;margin-bottom:16px">
    <span style="font-size:1.8rem;font-weight:900;color:#0a66c2;letter-spacing:-1px;line-height:1">in</span>
    <span style="color:#666;font-size:0.85rem;border-left:1px solid #e0e0e0;padding-left:16px">
        Smart Interview Prep &nbsp;
        <span style="background:#0a66c2;color:white;border-radius:4px;padding:2px 8px;font-size:0.7rem">AI ✨</span>
    </span>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────
def init():
    defaults = {
        "step": 1,
        "job_desc": "", "job_title": "", "company": "",
        "keywords": [], "all_questions": [], "questions": {},
        "current_q_idx": 0, "session_log": [], "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

def score_color(s):
    if s >= 75: return "#057642"
    if s >= 50: return "#d97706"
    return "#dc2626"

# ── STEP INDICATOR ───────────────────────────────────────────
steps_html = ""
labels = ["1 · Setup", "2 · Practice", "3 · Report"]
for i, label in enumerate(labels):
    cls = "active" if st.session_state.step == i+1 else ("done" if st.session_state.step > i+1 else "step")
    steps_html += f'<div class="step {cls}">{label}</div>'
st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# STEP 1 — SETUP
# ════════════════════════════════════════════════════════════
if st.session_state.step == 1:

    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">Prepare for your interview</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#666;font-size:0.9rem;margin-bottom:20px">Paste the job description from LinkedIn — we\'ll generate questions tailored to this specific role.</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        job_title = c1.text_input("Job Title", placeholder="e.g. Senior Data Scientist", label_visibility="visible")
        company   = c2.text_input("Company",   placeholder="e.g. Google", label_visibility="visible")

        job_desc = st.text_area(
            "Job Description",
            placeholder="Paste the full job description here...",
            height=220,
        )

        col_btn, col_info = st.columns([1, 3])
        start = col_btn.button("Generate Questions →", type="primary",
                               disabled=not job_desc.strip(), use_container_width=True)
        if not job_desc.strip():
            col_info.markdown('<p style="color:#999;font-size:0.85rem;margin-top:8px">Paste a job description to get started</p>', unsafe_allow_html=True)
        api_key = ""

        st.markdown('</div>', unsafe_allow_html=True)

        if start and job_desc.strip():
            import os
            # Load API key from Streamlit secrets (set in Streamlit Cloud dashboard)
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                st.session_state.api_key = api_key

            with st.spinner("Analysing job description..."):
                keywords = extract_keywords(job_desc, top_n=12)
                st.session_state.keywords  = keywords
                st.session_state.job_desc  = job_desc
                st.session_state.job_title = job_title
                st.session_state.company   = company

            demo_questions = {
                "role_summary": job_title,
                "behavioral": [
                    "Tell me about a time you had to learn a new skill quickly to deliver a project.",
                    "Describe a situation where you disagreed with a team member. How did you handle it?",
                    f"Give an example of how you used data to solve a real business problem.",
                ],
                "technical": [
                    f"Explain your experience with {keywords[0] if keywords else 'the main technology in this role'}.",
                    "How do you ensure quality and reliability in your work?",
                    "Walk me through how you would approach a complex analytical problem from scratch.",
                ]
            }
            with st.spinner("Generating your personalised questions..."):
                if api_key:
                    try:
                        questions = generate_questions(job_desc, keywords, num_behavioral=3, num_technical=3)
                    except Exception:
                        questions = demo_questions
                else:
                    questions = demo_questions

            all_q = []
            for q in questions.get("behavioral", []):
                all_q.append({"question": q, "type": "Behavioral"})
            for q in questions.get("technical", []):
                all_q.append({"question": q, "type": "Technical"})

            st.session_state.questions = questions
            st.session_state.all_questions = all_q
            st.session_state.step = 2
            st.rerun()

    with col_side:
        if st.session_state.keywords:
            st.markdown('<div class="li-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:700;font-size:0.85rem;margin-bottom:8px">Detected Skills</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="skill-chip">{k}</span>' for k in st.session_state.keywords)
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# STEP 2 — PRACTICE
# ════════════════════════════════════════════════════════════
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

    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        # Progress bar
        st.progress(idx / total)
        st.markdown(f'<p style="font-size:0.8rem;color:#666;margin-bottom:16px">Question {idx+1} of {total} · {st.session_state.job_title or "Your Role"}{" at " + st.session_state.company if st.session_state.company else ""}</p>', unsafe_allow_html=True)

        # Question card
        beh_s = "background:#0a66c2;color:white" if q_type == "Behavioral" else "background:#f0f0f0;color:#aaa"
        tec_s = "background:#057642;color:white" if q_type == "Technical" else "background:#f0f0f0;color:#aaa"
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-bottom:12px">
            <div style="padding:6px 16px;border-radius:999px;font-size:0.8rem;font-weight:600;{beh_s}">🤝 Behavioural</div>
            <div style="padding:6px 16px;border-radius:999px;font-size:0.8rem;font-weight:600;{tec_s}">⚙️ Technical</div>
        </div>
        <div class="q-card"><div class="q-text">{q_text}</div></div>
        """, unsafe_allow_html=True)

        # STAR tip for behavioral
        if q_type == "Behavioral":
            with st.expander("💡 STAR Method tip"):
                st.markdown("""
**Situation** → Set the context  
**Task** → What was your responsibility  
**Action** → Specific steps you took  
**Result** → Measurable outcome
                """)

        # Answer input
        answer = st.text_area(
            "Your answer",
            placeholder="Type your answer here... Aim for 80–150 words.",
            height=160,
            key=f"ans_{idx}",
            label_visibility="visible"
        )

        if answer:
            wc = len(answer.split())
            wc_cls = "good" if 80 <= wc <= 200 else "warn" if wc > 30 else ""
            st.markdown(f'<div class="word-count {wc_cls}">{wc} words {"✓" if 80 <= wc <= 200 else ""}</div>', unsafe_allow_html=True)

        # Store answer in session state to survive reruns
        if f"submitted_{idx}" not in st.session_state:
            st.session_state[f"submitted_{idx}"] = False

        bc1, bc2 = st.columns([1, 4])
        submit = bc1.button("Submit →", type="primary", 
                           disabled=not (answer and answer.strip()) or st.session_state[f"submitted_{idx}"],
                           key=f"submit_btn_{idx}")
        skip   = bc2.button("Skip question", key=f"skip_btn_{idx}")

        # After submit
        if submit and answer.strip() and not st.session_state[f"submitted_{idx}"]:
            st.session_state[f"submitted_{idx}"] = True
            with st.spinner("Scoring..."):
                scores = score_answer(q_text, answer, st.session_state.job_desc, q_type)

            ov = scores.get("overall", 0)
            c  = score_color(ov)
            pred = scores.get("pred_label", "")
            emoji = {"weak": "🔴", "good": "🟡", "strong": "🟢"}.get(pred, "")

            pred_label = pred.capitalize() if pred else "–"
            pred_desc = {"Weak": "Short or vague — add more detail", "Good": "Solid answer — add specific results", "Strong": "Excellent — clear, structured, specific"}.get(pred_label, "")
            if scores.get("mode") == "ml_model":
                st.markdown(f"""
                <div style="background:#fff;border-radius:8px;border:1px solid #e0e0e0;padding:20px 24px;margin:16px 0">
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                        <div style="font-size:2.5rem;font-weight:900;color:{c}">{ov}<span style="font-size:1rem;color:#999;font-weight:400">/100</span></div>
                        <div>
                            <div style="font-weight:700;font-size:1rem;color:{c}">{emoji} {pred_label}</div>
                            <div style="font-size:0.8rem;color:#666;margin-top:2px">{pred_desc}</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:8px">
                        <div style="flex:1;background:#f8f9fa;border-radius:6px;padding:10px;text-align:center">
                            <div style="font-size:1.2rem;font-weight:700;color:{score_color(scores.get('ml_score',0))}">{scores.get('ml_score',0)}/100</div>
                            <div style="font-size:0.72rem;color:#777;margin-top:2px">Answer quality<br><span style="color:#aaa">(structure &amp; content)</span></div>
                        </div>
                        <div style="flex:1;background:#f8f9fa;border-radius:6px;padding:10px;text-align:center">
                            <div style="font-size:1.2rem;font-weight:700;color:{score_color(scores.get('keyword_hit',0))}">{scores.get('keyword_hit',0)}/100</div>
                            <div style="font-size:0.72rem;color:#777;margin-top:2px">Skill relevance<br><span style="color:#aaa">(matches job requirements)</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-num" style="color:{c}">{ov}</div>
                        <div class="score-lbl">Overall</div>
                    </div>
                    <div class="score-box">
                        <div class="score-num" style="color:{score_color(scores.get('relevance',0))}">{scores.get('relevance',0)}</div>
                        <div class="score-lbl">Relevance</div>
                    </div>
                    <div class="score-box">
                        <div class="score-num" style="color:{score_color(scores.get('completeness',0))}">{scores.get('completeness',0)}</div>
                        <div class="score-lbl">Completeness</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.spinner("Getting AI feedback..."):
                if st.session_state.api_key:
                    try:
                        feedback = get_feedback(q_text, answer, q_type, scores, st.session_state.job_desc)
                    except Exception:
                        feedback = "**Tip:** Use the STAR method — Situation, Task, Action, Result. Include specific measurable outcomes to strengthen your answer."
                else:
                    feedback = "**Tip:** Use the STAR method — Situation, Task, Action, Result. Include specific measurable outcomes to strengthen your answer."

            st.markdown(f'<div class="feedback-box">{feedback}</div>', unsafe_allow_html=True)

            st.session_state.session_log.append({
                "question": q_text, "type": q_type,
                "answer": answer, "scores": scores, "feedback": feedback,
            })
            # Store scores/feedback so next button works after rerun
            st.session_state[f"scores_{idx}"] = scores
            st.session_state[f"feedback_{idx}"] = feedback

        # Show next button if this question was already submitted
        if st.session_state.get(f"submitted_{idx}", False):
            st.divider()
            label = "Next Question →" if idx + 1 < total else "See My Results →"
            if st.button(label, type="primary", key=f"next_{idx}"):
                st.session_state.current_q_idx = idx + 1
                st.rerun()

        if skip:
            st.session_state.session_log.append({
                "question": q_text, "type": q_type,
                "answer": "[Skipped]", "scores": {"overall": 0}, "feedback": "Skipped.",
            })
            st.session_state.current_q_idx += 1
            st.rerun()

    with col_side:
        # Progress tracker — only show if there are answered questions
        if log:
            st.markdown(f'<div style="font-weight:700;font-size:0.85rem;margin-bottom:10px;color:#1a1a1a">Progress · {len(log)}/{total} answered</div>', unsafe_allow_html=True)
            for i, e in enumerate(log):
                ov = e["scores"].get("overall", 0)
                c  = score_color(ov)
                pred = e["scores"].get("pred_label", "")
                emoji = {"weak":"🔴","good":"🟡","strong":"🟢"}.get(pred,"")
                st.markdown(f"""
                <div style="background:#fff;border-radius:6px;padding:8px 12px;margin-bottom:6px;
                            border:1px solid #e0e0e0;border-left:3px solid {c}">
                    <span style="font-size:0.72rem;color:#666">Q{i+1} · {e['type']}</span><br>
                    <span style="font-weight:700;color:{c}">{ov}/100</span>
                    <span style="float:right">{emoji}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Job overview — key skills + context
        job_title = st.session_state.job_title or "This Role"
        company   = st.session_state.company or ""
        keywords  = st.session_state.keywords[:8]
        header = f"{job_title}{' at ' + company if company else ''}"
        chips = "".join(f'<span class="skill-chip">{k}</span>' for k in keywords)
        st.markdown(f"""
        <div style="background:#fff;border-radius:8px;border:1px solid #e0e0e0;padding:16px">
            <div style="font-weight:700;font-size:0.85rem;color:#1a1a1a;margin-bottom:4px">{header}</div>
            <div style="font-size:0.75rem;color:#666;margin-bottom:10px">Key requirements</div>
            {chips}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# STEP 3 — REPORT
# ════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    log      = st.session_state.session_log
    answered = [e for e in log if e["answer"] != "[Skipped]"]

    if not answered:
        st.warning("No questions answered.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            init(); st.rerun()
        st.stop()

    avg = round(sum(e["scores"]["overall"] for e in answered) / len(answered))
    if avg >= 80:   grade, gc = "Excellent", "#057642"
    elif avg >= 65: grade, gc = "Strong", "#0a66c2"
    elif avg >= 50: grade, gc = "Good", "#d97706"
    else:           grade, gc = "Needs Work", "#dc2626"

    title_str = f'{st.session_state.job_title or "Your Role"}{" at " + st.session_state.company if st.session_state.company else ""}'

    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        # Header result
        st.markdown(f"""
        <div class="li-card" style="text-align:center;padding:32px">
            <div style="font-size:0.85rem;color:#666;margin-bottom:8px">{title_str}</div>
            <div style="font-size:3rem;font-weight:900;color:{gc}">{avg}<span style="font-size:1.2rem;color:#999">/100</span></div>
            <div style="font-size:1.1rem;font-weight:700;color:{gc};margin-top:4px">{grade}</div>
            <div style="font-size:0.82rem;color:#999;margin-top:8px">{len(answered)} questions answered</div>
        </div>
        """, unsafe_allow_html=True)

        # AI Summary
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div class="li-section-title">AI Coach Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating summary..."):
            summary = get_session_summary(answered) if st.session_state.api_key else "Add your API key to get a personalised session summary with strengths and improvement areas."
        st.markdown(f'<div class="feedback-box">{summary}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Question review
        st.markdown('<div class="li-section-title" style="margin-top:8px">Question Review</div>', unsafe_allow_html=True)
        tab_all, tab_beh, tab_tech = st.tabs(["All", "Behavioural", "Technical"])

        def render_entries(entries):
            for i, e in enumerate(entries):
                ov   = e["scores"].get("overall", 0)
                pred = e["scores"].get("pred_label", "")
                emoji= {"weak":"🔴","good":"🟡","strong":"🟢"}.get(pred,"")
                c    = score_color(ov)
                with st.expander(f"Q{i+1} · {e['type']} · {emoji} {ov}/100 — {e['question'][:60]}..."):
                    st.markdown(f"**Question:** {e['question']}")
                    st.markdown(f"**Your answer:** {e['answer']}")
                    st.markdown(f'**Score: <span style="color:{c};font-weight:700">{ov}/100</span>**', unsafe_allow_html=True)
                    st.markdown(f'<div class="feedback-box">{e["feedback"]}</div>', unsafe_allow_html=True)

        with tab_all:  render_entries(answered)
        with tab_beh:  render_entries([e for e in answered if e["type"]=="Behavioral"])
        with tab_tech: render_entries([e for e in answered if e["type"]=="Technical"])

    with col_side:
        # Score breakdown
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700;font-size:0.85rem;margin-bottom:12px">Score Breakdown</div>', unsafe_allow_html=True)

        mode = answered[0]["scores"].get("mode","heuristic")
        if mode == "ml_model":
            labels_q = [e["scores"].get("pred_label","") for e in answered]
            strong = labels_q.count("strong")
            good   = labels_q.count("good")
            weak   = labels_q.count("weak")
            st.markdown(f"""
            <div style="font-size:0.82rem;line-height:2">
                🟢 Strong: <b>{strong}</b><br>
                🟡 Good: <b>{good}</b><br>
                🔴 Weak: <b>{weak}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            import pandas as pd
            chart_df = pd.DataFrame({"Score": [
                round(sum(e["scores"].get("relevance",0) for e in answered)/len(answered)),
                round(sum(e["scores"].get("completeness",0) for e in answered)/len(answered)),
                round(sum(e["scores"].get("keyword_hit",0) for e in answered)/len(answered)),
            ]}, index=["Relevance","Completeness","Keywords"])
            st.bar_chart(chart_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Skills
        st.markdown('<div class="li-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700;font-size:0.85rem;margin-bottom:8px">Role Skills</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="skill-chip">{k}</span>' for k in st.session_state.keywords[:8])
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download + restart
        st.markdown('<div style="margin-top:8px">', unsafe_allow_html=True)
        report = json.dumps({"job_title": st.session_state.job_title, "company": st.session_state.company,
                             "grade": grade, "avg_overall": avg, "questions": answered}, indent=2)
        st.download_button("⬇️ Download Report", data=report,
                           file_name="interview_report.json", mime="application/json",
                           use_container_width=True)
        if st.button("Start New Interview", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            init(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

