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


def build_candidate_mirror(answers, job_title, keywords):
    """
    Synthesises ALL answers together in one LLM call to produce a
    'Candidate Mirror' — how the interviewer perceived the candidate.
    This is different from per-answer feedback: it looks for patterns
    across the whole session.
    """
    c = _client()
    kw = ", ".join(keywords[:8]) if keywords else "N/A"

    session_text = ""
    for i, a in enumerate(answers, 1):
        if a.get("answer") and a["answer"] != "[skipped]":
            session_text += f"\nQ{i} ({a.get('type','')}) : {a['question']}\nAnswer: {a['answer']}\n"

    prompt = f"""You are a senior hiring manager who has just finished interviewing a candidate for {job_title}.
Required skills: {kw}

Here are all their answers:
{session_text}

Analyse the candidate's answers as a whole — not per question, but as a complete picture.
Look for patterns, recurring strengths, consistent gaps, and the overall impression they leave.

Return ONLY this JSON:
{{
  "communication_style": "2 sentences describing how this person communicates — are they data-driven, storytelling, direct, vague, structured, etc.",
  "blind_spots": ["pattern they consistently avoid, e.g. never quantifies outcomes", "another recurring gap"],
  "how_you_come_across": "Write this in first person as if the interviewer is describing the candidate to a colleague after the interview. 3-4 sentences. Be honest.",
  "interviewer_doubts": ["a doubt the interviewer is left with based on the answers", "another doubt", "a third doubt"],
  "hidden_strength": "One genuine strength that came through consistently that the candidate probably does not realise is impressive"
}}"""

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def build_prep_plan(answers, job_title, keywords, competency_averages):
    """
    Generates a personalised 7-day prep plan based on the candidate's
    weakest competency dimensions and the job requirements.
    The plan is grounded in the actual scores — not generic advice.
    """
    c = _client()
    kw = ", ".join(keywords[:8]) if keywords else "N/A"

    # find the two weakest dims to focus the plan on
    dims = ["structure", "specificity", "impact", "relevance", "communication"]
    sorted_dims = sorted(dims, key=lambda d: competency_averages.get(d, 50))
    weakest_two = sorted_dims[:2]
    scores_str = ", ".join(f"{d}: {competency_averages.get(d, 50)}/100" for d in dims)

    prompt = f"""You are an interview coach building a personalised preparation plan.

Role: {job_title}
Key skills required: {kw}
Candidate's competency scores: {scores_str}
Two dimensions that need most work: {', '.join(weakest_two)}

Build a focused 7-day interview prep plan that targets the weak dimensions and the role requirements.
Each day should have one clear, specific action — not vague advice like "practice more."

Return ONLY this JSON:
{{
  "focus_areas": ["the two main things this plan targets"],
  "days": [
    {{"day": 1, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 2, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 3, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 4, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 5, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 6, "title": "short title", "task": "specific actionable task in 2 sentences"}},
    {{"day": 7, "title": "short title", "task": "specific actionable task in 2 sentences"}}
  ],
  "key_reminder": "One sentence the candidate should read the morning of the interview"
}}"""

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
