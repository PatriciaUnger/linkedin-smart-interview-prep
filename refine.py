"""
refine.py
─────────
The "Refine your answer" feature for V3.

When a candidate is unhappy with a score, they can rewrite the answer and see
a side-by-side before/after comparison. This turns the app from a one-shot
assessment into an iterative practice tool — the original V2 documentation
explicitly flagged this as future work.

How it works:
    1. Candidate submits a revised answer for a specific question
    2. We run the revised answer through the same scoring + coaching pipeline
       that produced the original feedback (ai_coach.analyse_answer)
    3. We compute per-dimension deltas (new - old)
    4. We make ONE LLM call to produce a short, specific "what changed" note
       that references the actual delta numbers — not generic "nice work"

Storage:
    Each answer in st.session_state['answers'] grows a 'revisions' list.
    Each revision has the same shape as the original entry plus a delta dict.
    This means downstream features (Candidate Mirror, Prep Plan) can either
    use the original OR the latest revision by flipping one parameter later.
"""

import os
import json
import anthropic
import streamlit as st

from ai_coach import analyse_answer, COMPETENCY_DIMS


def _client():
    key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    return anthropic.Anthropic(api_key=key)


def _compute_deltas(old_scores: dict, new_scores: dict) -> dict:
    """Per-dimension change. Positive = improvement."""
    deltas = {}
    for d in COMPETENCY_DIMS:
        old_v = old_scores.get(d, 0)
        new_v = new_scores.get(d, 0)
        deltas[d] = new_v - old_v
    return deltas


def _what_changed(question: str, old_answer: str, new_answer: str,
                  deltas: dict, overall_delta: int) -> str:
    """
    One LLM call to describe the specific change. Short and grounded in
    the actual delta numbers, not generic praise.
    """
    biggest_improvement = max(deltas, key=deltas.get)
    biggest_regression = min(deltas, key=deltas.get)

    prompt = f"""A candidate revised their interview answer. Describe what they changed
and why the new score is different. Be specific — reference the actual edits, not
generic interview advice.

Question: {question}

Original answer: {old_answer}

Revised answer: {new_answer}

Score changes (positive = improved):
- Structure: {deltas['structure']:+d}
- Specificity: {deltas['specificity']:+d}
- Impact: {deltas['impact']:+d}
- Relevance: {deltas['relevance']:+d}
- Communication: {deltas['communication']:+d}
- Overall: {overall_delta:+d}

Biggest improvement: {biggest_improvement}
Biggest regression (if any): {biggest_regression}

Write 2 sentences:
- Sentence 1: what concretely changed in the answer (reference specific phrases or
  structural changes). Do not say "the candidate" — say "you".
- Sentence 2: why that change moved the score on the biggest-impact dimension.

If the overall score went DOWN, be honest about it — do not pretend it was an
improvement. If the score barely changed, say the edits were cosmetic and suggest
what a meaningful edit would have looked like.

Return only a JSON object:
{{"summary": "..."}}"""

    c = _client()
    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("summary", "")
    except Exception:
        return raw[:300]


def refine_answer(answer_idx: int, revised_text: str,
                  job_title: str, keywords: list,
                  rag_engine) -> dict:
    """
    Re-run the scoring + coaching pipeline on a revised answer.
    Returns a revision dict; does NOT mutate session state (caller does that).

    Parameters
    ----------
    answer_idx : position of the target answer in st.session_state['answers']
    revised_text : the candidate's new answer text
    job_title, keywords : passed through to the existing coach
    rag_engine : the shared RAGEngine instance

    Returns
    -------
    dict with the same shape as an entry in session_state['answers'], plus
    'deltas' (per-dim change) and 'summary_of_change' (LLM-generated note).
    """
    answers = st.session_state["answers"]
    original = answers[answer_idx]

    # If the candidate has already revised before, compare against the latest
    # revision, not the original. This makes iterative refinement meaningful.
    prior = original
    if original.get("revisions"):
        prior = original["revisions"][-1]

    question = original["question"]
    q_type = original.get("type", "Behavioral")

    # Re-run the full V2 scoring + coaching pipeline
    rag_context = rag_engine.build_rag_context(revised_text, k=2)
    analysis = analyse_answer(
        question=question,
        answer=revised_text,
        question_type=q_type,
        job_title=job_title,
        keywords=keywords,
        rag_context=rag_context,
    )

    new_scores = analysis.get("competencies", {d: 50 for d in COMPETENCY_DIMS})
    old_scores = prior.get("competencies", {d: 50 for d in COMPETENCY_DIMS})

    # Overall is the average of the 5 competency scores (mirrors how the
    # original scoring works — keeps numbers comparable).
    new_overall = round(sum(new_scores.get(d, 0) for d in COMPETENCY_DIMS) / len(COMPETENCY_DIMS))
    old_overall = prior.get("score", 50)
    overall_delta = new_overall - old_overall

    deltas = _compute_deltas(old_scores, new_scores)

    try:
        change_summary = _what_changed(
            question=question,
            old_answer=prior["answer"],
            new_answer=revised_text,
            deltas=deltas,
            overall_delta=overall_delta,
        )
    except Exception:
        change_summary = (
            f"Overall changed by {overall_delta:+d} points. "
            f"Biggest shift: {max(deltas, key=deltas.get)} ({deltas[max(deltas, key=deltas.get)]:+d})."
        )

    revision = {
        "answer": revised_text,
        "score": new_overall,
        "feedback": analysis.get("feedback", ""),
        "strengths": analysis.get("strengths", []),
        "improvements": analysis.get("improvements", []),
        "competencies": new_scores,
        "rag_context_used": rag_context,
        # Revision-specific fields:
        "deltas": deltas,
        "overall_delta": overall_delta,
        "summary_of_change": change_summary,
        "compared_against": "previous_revision" if original.get("revisions") else "original",
    }
    return revision
