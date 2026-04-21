"""
eval_harness.py
───────────────
The evaluation harness for V3.

Core function: run a single test case through a 2x2 matrix of variants
    {tfidf, semantic} × {v1_single_prompt, v2_two_step}
and return a structured comparison.

Why 2x2?
    My V2 feedback flagged two separate "to improve" items:
      (a) swap TF-IDF for semantic embeddings
      (b) run a user study to measure if V2 is actually better than V1
    A 2x2 matrix lets me isolate which change matters. I can answer:
      - Did the embeddings change which KB examples get surfaced?
      - Did the two-step feedback produce more stable scores than single-prompt?
      - Are these two effects additive, or does one dominate?

Variants:
    - Retrieval:
        tfidf     → rag_engine.RAGEngine(docs, mode='tfidf')
        semantic  → rag_engine.RAGEngine(docs, mode='semantic')
    - Feedback style:
        v1_single → one LLM call that scores AND coaches in the same prompt
                    (the original V1 approach before I discovered scoring drift)
        v2_two_step → score_competencies(), then get_coaching() with scores
                      and retrieved examples — the V2 approach

Output:
    A dict with:
      - test_case_id, timestamp, model name
      - For each of the 4 variants:
          - top_3_retrievals (doc id + similarity)
          - feedback (text)
          - scores (5 dims)
          - any errors
      - Cross-variant comparisons:
          - retrieval_overlap: Jaccard of top-3 sets between tfidf and semantic
          - score_divergence: mean abs diff per dim between v1 and v2 styles
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import anthropic

from rag_engine import RAGEngine
from interview_kb import get_knowledge_base
from ai_coach import score_competencies, get_coaching, COMPETENCY_DIMS


MODEL_NAME = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────────────────────────────────────
# Engine cache — build once per harness run, reuse across test cases.
# ─────────────────────────────────────────────────────────────────────────────
_engine_cache = {}


def _get_engine(mode: str) -> RAGEngine:
    if mode not in _engine_cache:
        _engine_cache[mode] = RAGEngine(get_knowledge_base(), mode=mode)
    return _engine_cache[mode]


def _client():
    """Get the Anthropic client. Works both inside Streamlit and standalone."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        # Fall back to Streamlit secrets if available (when run inside app)
        try:
            import streamlit as st
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return anthropic.Anthropic(api_key=key)


# ─────────────────────────────────────────────────────────────────────────────
# V1-style single-prompt feedback (the V1 baseline we're comparing against)
# ─────────────────────────────────────────────────────────────────────────────
def _v1_single_prompt_feedback(question: str, answer: str, q_type: str,
                                rag_context: str) -> dict:
    """
    Reproduces the original V1 approach: score AND coach in one LLM call.

    This is the approach I moved away from in V2 after discovering that the
    model was rationalising scores to match whatever feedback it had already
    written. Documenting the original behaviour here lets me empirically show
    the difference between the two approaches in the same run.
    """
    c = _client()
    prompt = f"""You are an interview coach. Score this answer on five dimensions (0-100)
and provide coaching feedback.

Question ({q_type}): {question}
Answer: {answer}

{rag_context}

Return ONLY this JSON:
{{
  "structure": <0-100>,
  "specificity": <0-100>,
  "impact": <0-100>,
  "relevance": <0-100>,
  "communication": <0-100>,
  "feedback": "2-3 sentences of coaching",
  "strengths": ["..."],
  "improvements": ["..."]
}}"""

    r = c.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Run a single variant (one retrieval mode + one feedback style)
# ─────────────────────────────────────────────────────────────────────────────
def _run_variant(test_case: dict, retrieval_mode: str, feedback_style: str) -> dict:
    """
    Run one test case through one variant.
    Returns a dict with retrievals, scores, feedback, timing, and errors if any.
    """
    t0 = time.time()
    out = {
        "retrieval_mode": retrieval_mode,
        "feedback_style": feedback_style,
        "top_3_retrievals": [],
        "scores": {},
        "feedback": "",
        "strengths": [],
        "improvements": [],
        "elapsed_sec": None,
        "error": None,
    }

    try:
        engine = _get_engine(retrieval_mode)

        # Get top-3 strong retrievals (mirrors what build_rag_context does)
        top3 = engine.retrieve_raw(test_case["answer"], k=3, quality=["strong"])
        out["top_3_retrievals"] = [
            {
                "skill": d.get("skill", ""),
                "quality": d.get("quality", ""),
                "similarity": round(d.get("similarity", 0.0), 4),
                "text_preview": d.get("text", "")[:100] + "...",
            }
            for d in top3
        ]

        rag_context = engine.build_rag_context(test_case["answer"], k=2)

        if feedback_style == "v1_single":
            r = _v1_single_prompt_feedback(
                test_case["question"],
                test_case["answer"],
                test_case["question_type"],
                rag_context,
            )
            out["scores"] = {d: r.get(d, 0) for d in COMPETENCY_DIMS}
            out["feedback"] = r.get("feedback", "")
            out["strengths"] = r.get("strengths", [])
            out["improvements"] = r.get("improvements", [])

        elif feedback_style == "v2_two_step":
            scores = score_competencies(
                test_case["question"],
                test_case["answer"],
                test_case["question_type"],
            )
            out["scores"] = {d: scores.get(d, 0) for d in COMPETENCY_DIMS}

            coaching = get_coaching(
                test_case["question"],
                test_case["answer"],
                test_case["question_type"],
                test_case["job_title"],
                keywords=[],
                rag_context=rag_context,
                scores=scores,
            )
            out["feedback"] = coaching.get("feedback", "")
            out["strengths"] = coaching.get("strengths", [])
            out["improvements"] = coaching.get("improvements", [])

        else:
            raise ValueError(f"Unknown feedback_style: {feedback_style}")

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-variant analysis
# ─────────────────────────────────────────────────────────────────────────────
def _retrieval_overlap(retrievals_a: list, retrievals_b: list) -> dict:
    """Jaccard-style overlap between two top-3 retrieval lists."""
    def key_of(r):
        return (r.get("skill", ""), r.get("text_preview", "")[:40])
    set_a = {key_of(r) for r in retrievals_a}
    set_b = {key_of(r) for r in retrievals_b}
    if not set_a and not set_b:
        return {"overlap_count": 0, "jaccard": 0.0}
    intersection = set_a & set_b
    union = set_a | set_b
    return {
        "overlap_count": len(intersection),
        "jaccard": round(len(intersection) / len(union), 3) if union else 0.0,
    }


def _score_divergence(scores_a: dict, scores_b: dict) -> dict:
    """Mean absolute difference per dimension between two score dicts."""
    diffs = {}
    total = 0
    n = 0
    for d in COMPETENCY_DIMS:
        a = scores_a.get(d)
        b = scores_b.get(d)
        if a is None or b is None:
            continue
        diffs[d] = abs(a - b)
        total += diffs[d]
        n += 1
    return {
        "per_dim_abs_diff": diffs,
        "mean_abs_diff": round(total / n, 2) if n else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(test_case: dict, variants: list = None) -> dict:
    """
    Run a single test case through all 4 variants (or a subset) and compute
    cross-variant comparisons. Returns a structured result ready to be saved.

    Parameters
    ----------
    test_case : dict from eval_testset.TEST_SET
    variants : list of (retrieval_mode, feedback_style) tuples, or None for all 4

    Returns
    -------
    dict with: metadata, per-variant results, cross-variant comparisons
    """
    if variants is None:
        variants = [
            ("tfidf",    "v1_single"),
            ("tfidf",    "v2_two_step"),
            ("semantic", "v1_single"),
            ("semantic", "v2_two_step"),
        ]

    run_id = uuid.uuid4().hex[:8]
    started = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    variant_results = {}
    for retrieval_mode, feedback_style in variants:
        key = f"{retrieval_mode}__{feedback_style}"
        variant_results[key] = _run_variant(test_case, retrieval_mode, feedback_style)

    # Cross-variant comparisons — useful when all 4 variants were run
    comparisons = {}

    # Compare retrieval modes (holding feedback style constant)
    if ("tfidf__v2_two_step" in variant_results
        and "semantic__v2_two_step" in variant_results):
        comparisons["retrieval_overlap_v2"] = _retrieval_overlap(
            variant_results["tfidf__v2_two_step"]["top_3_retrievals"],
            variant_results["semantic__v2_two_step"]["top_3_retrievals"],
        )

    # Compare feedback styles (holding retrieval constant)
    if ("semantic__v1_single" in variant_results
        and "semantic__v2_two_step" in variant_results):
        comparisons["score_divergence_v1_vs_v2_semantic"] = _score_divergence(
            variant_results["semantic__v1_single"]["scores"],
            variant_results["semantic__v2_two_step"]["scores"],
        )

    return {
        "run_id": run_id,
        "timestamp_utc": started,
        "model": MODEL_NAME,
        "test_case": {
            "id": test_case["id"],
            "question": test_case["question"],
            "answer": test_case["answer"],
            "job_title": test_case["job_title"],
            "question_type": test_case["question_type"],
            "expected_skill": test_case.get("expected_skill"),
            "expected_quality_tier": test_case.get("expected_quality_tier"),
            "notes": test_case.get("notes", ""),
        },
        "variants": variant_results,
        "comparisons": comparisons,
    }
