import os
import anthropic
import streamlit as st

SYSTEM = """You are an interview coach on LinkedIn Smart Interview Prep.
You have access to the user's full mock interview session below.

Your job:
- Answer questions about their specific answers and scores
- Give targeted advice on how to improve weak answers
- Show example strong answers using STAR when asked
- Be honest but encouraging
- Keep replies short (3-5 sentences) unless showing an example answer
- Always tie your advice back to the specific job and session data

Don't make up questions or scores that aren't in the session."""


def _session_summary(data):
    lines = [
        f"Role: {data.get('job_title', '?')} at {data.get('company', '?')}",
        f"Key skills: {', '.join(data.get('keywords', [])[:8])}",
        f"Overall score: {data.get('overall_score', '?')}/100",
        "",
        "Questions and answers:",
    ]
    for i, a in enumerate(data.get("answers", []), 1):
        lines += [
            f"\nQ{i}: {a.get('question', '')}",
            f"Type: {a.get('type', '')}",
            f"Answer: {a.get('answer', '[skipped]')}",
            f"Score: {a.get('score', '?')}/100",
            f"Feedback: {a.get('feedback', '')}",
        ]
    return "\n".join(lines)


def opening_message(data):
    score = data.get("overall_score", 0)
    job = data.get("job_title", "this role")
    answers = data.get("answers", [])
    weak = [a for a in answers if a.get("score", 100) < 50]

    if score >= 75:
        msg = f"Nice work — {score}/100 for {job}."
    elif score >= 50:
        msg = f"Solid session, {score}/100 for {job}. Some clear areas to work on."
    else:
        msg = f"You scored {score}/100 for {job}. Let's figure out what to improve."

    if weak:
        msg += f" I noticed your answer on '{weak[0].get('question', '')[:55]}...' had room to grow. Want me to walk through it?"
    else:
        msg += " Ask me anything about your answers or what to prepare next."
    return msg


def chat(user_message, history, session_data):
    key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    c = anthropic.Anthropic(api_key=key)

    system_with_context = SYSTEM + "\n\n" + _session_summary(session_data)
    history.append({"role": "user", "content": user_message})

    r = c.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=system_with_context,
        messages=history,
    )
    reply = r.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply
