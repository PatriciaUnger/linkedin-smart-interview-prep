import os
import json
import anthropic
import streamlit as st

# the five dimensions I want to track per answer
COMPETENCY_DIMS = ["structure", "specificity", "impact", "relevance", "communication"]


def _client():
    key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    return anthropic.Anthropic(api_key=key)


def generate_questions(job_title, company, job_description, difficulty, num_questions=5):
    c = _client()
    prompt = f"""You are a recruiter at {company} hiring for {job_title}.

Job description:
{job_description[:1200]}

Seniority: {difficulty}

Write exactly {num_questions} interview questions for this role. Mix behavioural and technical.
Return ONLY a JSON array, no other text:
[
  {{"question": "...", "type": "behavioural", "skill": "..."}},
  {{"question": "...", "type": "technical", "skill": "..."}}
]"""

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def score_competencies(question, answer, q_type):
    """
    First LLM call: just score, no coaching yet.
    I split this into a separate call because mixing scoring and feedback
    in one prompt gave inconsistent results — the model would justify
    scores with the feedback instead of scoring objectively.
    """
    c = _client()
    prompt = f"""Score this interview answer on five dimensions. Only score, no advice.

Question ({q_type}): {question}
Answer: {answer}

Dimensions (0-100 each):
- structure: logical flow, e.g. STAR format
- specificity: concrete details, numbers, examples vs. vague statements
- impact: measurable outcomes mentioned
- relevance: how directly it answers the question
- communication: clarity and conciseness

Return only this JSON:
{{
  "structure": 0,
  "specificity": 0,
  "impact": 0,
  "relevance": 0,
  "communication": 0,
  "weakest": "dimension_name",
  "strongest": "dimension_name"
}}"""

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def get_coaching(question, answer, q_type, job_title, keywords, rag_context, scores):
    """
    Second LLM call: generate feedback using the scores from the first call
    plus retrieved examples from the knowledge base.
    The scores tell the model where to focus; the examples give it something
    concrete to reference rather than producing generic advice.
    """
    c = _client()
    kw = ", ".join(keywords[:8]) if keywords else "N/A"

    if scores:
        score_summary = "\n".join(f"  {k}: {scores[k]}/100" for k in COMPETENCY_DIMS if k in scores)
        weakest = scores.get("weakest", "specificity")
        score_block = f"Dimension scores:\n{score_summary}\nFocus feedback on: {weakest}"
    else:
        score_block = ""

    prompt = f"""You are an interview coach. Give feedback on this answer using the scores and examples below.

Role: {job_title}
Required skills: {kw}
Question ({q_type}): {question}
Candidate answer: {answer}

{score_block}

{rag_context}

Write 2-3 sentences of specific coaching. Mention what dimension is weakest and reference
the strong example to show what good looks like. Be direct, not generic.

Return only this JSON:
{{
  "feedback": "...",
  "strengths": ["..."],
  "improvements": ["..."]
}}"""

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def analyse_answer(question, answer, question_type, job_title, keywords, rag_context):
    # step 1: score the answer across dimensions
    try:
        scores = score_competencies(question, answer, question_type)
    except Exception:
        scores = {d: 50 for d in COMPETENCY_DIMS}
        scores["weakest"] = "specificity"
        scores["strongest"] = "communication"

    # step 2: use those scores + retrieved examples to generate coaching
    try:
        coaching = get_coaching(
            question, answer, question_type,
            job_title, keywords, rag_context, scores
        )
    except Exception:
        coaching = {
            "feedback": "Decent attempt. Try to add specific numbers and outcomes to strengthen your answer.",
            "strengths": ["Relevant topic covered"],
            "improvements": ["Add measurable results"],
        }

    return {
        "feedback": coaching.get("feedback", ""),
        "strengths": coaching.get("strengths", []),
        "improvements": coaching.get("improvements", []),
        "competencies": {d: scores.get(d, 50) for d in COMPETENCY_DIMS},
    }
