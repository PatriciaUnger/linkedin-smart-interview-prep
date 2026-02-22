"""
ai_coach.py
───────────
All calls to the Anthropic Claude API live here.
The Streamlit app imports only generate_questions() and get_feedback().
"""

import os
import anthropic
import json
import re

# Initialise client – reads ANTHROPIC_API_KEY from environment
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client


MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for prototyping


# ─────────────────────────────────────────────────────────────
# 1. QUESTION GENERATION
# ─────────────────────────────────────────────────────────────

def generate_questions(
    job_description: str,
    keywords: list[str],
    num_behavioral: int = 3,
    num_technical: int = 3,
    difficulty: str = "Medium",
) -> dict:
    """
    Given a job description and extracted keywords, generate interview questions.

    Returns:
        {
          "behavioral": ["Q1", "Q2", ...],
          "technical":  ["Q1", "Q2", ...],
          "role_summary": "short 1-line summary of the role"
        }
    """
    keyword_str = ", ".join(keywords) if keywords else "not provided"

    prompt = f"""You are an expert technical recruiter. Based on the job description below, 
generate interview questions tailored specifically to this role.

JOB DESCRIPTION:
{job_description[:3000]}

KEY SKILLS DETECTED: {keyword_str}
DIFFICULTY LEVEL: {difficulty}

Generate exactly {num_behavioral} behavioral questions and {num_technical} technical questions.
Behavioral questions should use the STAR method format (Situation, Task, Action, Result).
Technical questions should test depth of knowledge relevant to this specific role.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "role_summary": "one sentence describing the role",
  "behavioral": ["question 1", "question 2", "question 3"],
  "technical": ["question 1", "question 2", "question 3"]
}}"""

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        # Validate structure
        result.setdefault("role_summary", "")
        result.setdefault("behavioral", [])
        result.setdefault("technical", [])
        return result
    except Exception as e:
        return {
            "role_summary": "Could not parse role",
            "behavioral": [f"Error generating questions: {e}"],
            "technical": [],
        }


# ─────────────────────────────────────────────────────────────
# 2. ANSWER FEEDBACK
# ─────────────────────────────────────────────────────────────

def get_feedback(
    question: str,
    answer: str,
    question_type: str,
    scores: dict,
    job_description: str = "",
) -> str:
    """
    Provide detailed, actionable coaching feedback on the candidate's answer.
    Returns a markdown-formatted string.
    """
    score_summary = (
        f"Relevance: {scores.get('relevance', 0)}/100 | "
        f"Completeness: {scores.get('completeness', 0)}/100 | "
        f"Keyword Usage: {scores.get('keyword_hit', 0)}/100 | "
        f"Overall: {scores.get('overall', 0)}/100"
    )

    jd_context = f"\nJOB CONTEXT: {job_description[:500]}" if job_description else ""

    prompt = f"""You are an expert interview coach giving feedback to a job candidate.

QUESTION ({question_type}): {question}

CANDIDATE'S ANSWER: {answer}

AUTOMATED SCORES: {score_summary}{jd_context}

Provide concise, encouraging yet honest coaching feedback. Structure your response as:

**What worked well ✅**
(2-3 specific strengths)

**Areas to improve 🔧**  
(2-3 concrete suggestions)

**Stronger answer example 💡**
(A brief 3-4 sentence model answer)

Keep the total response under 250 words. Be specific, actionable, and encouraging."""

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"⚠️ Could not generate AI feedback: {e}\n\nCheck your ANTHROPIC_API_KEY."


# ─────────────────────────────────────────────────────────────
# 3. SESSION SUMMARY
# ─────────────────────────────────────────────────────────────

def get_session_summary(session_data: list[dict]) -> str:
    """
    Given a list of {question, answer, scores} dicts, generate an overall
    coaching summary for the entire interview session.
    """
    if not session_data:
        return "No session data to summarise."

    qa_text = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}\nScore: {item['scores'].get('overall', 0)}/100"
        for item in session_data[:6]   # cap to avoid huge prompts
    )

    prompt = f"""You are a career coach. The candidate just completed a mock interview. 
Here are their Q&A pairs with scores:

{qa_text}

Write a concise overall performance summary (max 200 words) covering:
1. Overall performance level (Needs Work / Good / Strong / Excellent)
2. Top 2 strengths shown across all answers
3. Top 2 priority areas to practise before the real interview
4. One specific action they can take today to improve

Be encouraging and specific."""

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"⚠️ Could not generate summary: {e}"
