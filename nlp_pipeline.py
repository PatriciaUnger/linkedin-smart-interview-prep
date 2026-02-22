"""
nlp_pipeline.py
───────────────
Runtime NLP logic – imported by app.py.

Two scoring modes:
  1. ML model (preferred)  – loads artefacts/model_pipeline.pkl trained by train_model.py
  2. Heuristic fallback    – used if artefacts not found (first run before training)
"""

import re, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARTEFACT_DIR = Path(__file__).parent / "artefacts"
_cache = {}

def _load_model():
    if "pipeline" not in _cache:
        pipeline_path = ARTEFACT_DIR / "model_pipeline.pkl"
        encoder_path  = ARTEFACT_DIR / "label_encoder.pkl"
        if pipeline_path.exists() and encoder_path.exists():
            import joblib
            _cache["pipeline"]        = joblib.load(pipeline_path)
            _cache["le"]              = joblib.load(encoder_path)
            _cache["model_available"] = True
        else:
            _cache["model_available"] = False
    return _cache


_STOPWORDS = {
    "a","about","above","after","again","all","am","an","and","any","are","as",
    "at","be","been","being","but","by","can","did","do","does","each","for",
    "from","get","had","has","have","he","her","him","his","how","i","if","in",
    "into","is","it","its","me","my","no","not","of","off","on","or","our",
    "out","own","same","she","so","some","such","than","that","the","their",
    "them","then","there","these","they","this","those","to","too","up","us",
    "was","we","were","what","when","where","which","while","who","will","with",
    "would","you","your","also","etc","must","may","very","just","own"
}


# ─────────────────────────────────────────────────────────────
# 1. KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_keywords(text: str, top_n: int = 12) -> list:
    if not text or len(text.strip()) < 20:
        return []
    sentences = re.split(r"[.;\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 2:
        sentences = [text]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              max_features=200, token_pattern=r"[a-zA-Z]{3,}")
        matrix = vec.fit_transform(sentences)
        scores = matrix.sum(axis=0).A1
        vocab  = vec.get_feature_names_out()
        ranked = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked if w not in _STOPWORDS][:top_n]
    except Exception:
        from collections import Counter
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        freq   = Counter(t for t in tokens if t not in _STOPWORDS)
        return [w for w, _ in freq.most_common(top_n)]


# ─────────────────────────────────────────────────────────────
# 2a. ML MODEL SCORING
# ─────────────────────────────────────────────────────────────
def _score_with_model(question, answer, question_type, job_description):
    cache    = _load_model()
    words    = answer.split()
    wc       = len(words)
    sents    = [s.strip() for s in answer.replace("!",".").replace("?",".").split(".") if s.strip()]
    avg_sent = wc / max(len(sents), 1)
    star_kw  = ["situation","task","action","result","achieved","outcome",
                "reduced","increased","improved","delivered","percent","%"]

    row = pd.DataFrame([{
        "answer_text":       answer,
        "question_type":     question_type if question_type in ["Behavioral","Technical"] else "Behavioral",
        "word_count":        float(wc),
        "has_example":       int(any(k in answer.lower() for k in ["example","e.g.","instance"])),
        "has_numbers":       int(any(c.isdigit() for c in answer)),
        "uses_first_person": int(" i " in answer.lower() or answer.lower().startswith("i ")),
        "avg_sentence_len":  float(avg_sent),
        "answer_length_cat": "short" if wc < 30 else ("long" if wc > 100 else "medium"),
        "lexical_richness":  len(set(answer.lower().split())) / max(wc, 1),
        "star_score":        float(sum(1 for k in star_kw if k in answer.lower())),
        "sentence_count":    float(len(sents)),
        "is_very_short":     int(wc < 20),
    }])

    pipeline = cache["pipeline"]
    le       = cache["le"]
    proba    = pipeline.predict_proba(row)[0]
    classes  = list(le.classes_)

    label_weights = {"weak": 0, "good": 60, "strong": 100}
    ml_score  = int(sum(proba[i] * label_weights.get(classes[i], 50) for i in range(len(classes))))
    ml_score  = max(0, min(100, ml_score))
    pred_label= classes[int(np.argmax(proba))]
    confidence= int(max(proba) * 100)
    kw_hit    = _keyword_hit(answer, job_description)
    overall   = int(ml_score * 0.70 + kw_hit * 0.30)

    return {"ml_score": ml_score, "keyword_hit": kw_hit, "overall": overall,
            "pred_label": pred_label, "confidence": confidence, "mode": "ml_model"}


# ─────────────────────────────────────────────────────────────
# 2b. HEURISTIC FALLBACK
# ─────────────────────────────────────────────────────────────
def _score_heuristic(question, answer, job_description):
    try:
        vec    = TfidfVectorizer(stop_words="english", token_pattern=r"[a-zA-Z]{3,}")
        matrix = vec.fit_transform([question, answer])
        rel    = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        rel = 0.0
    relevance_score = min(int(rel * 250), 100)

    wc = len(answer.split())
    if wc < 20:       comp = wc / 20
    elif wc <= 200:   comp = 1.0
    else:             comp = max(0.5, 1 - (wc - 200) / 400)
    completeness_score = int(comp * 100)

    kw_hit  = _keyword_hit(answer, job_description)
    overall = int(relevance_score * 0.45 + completeness_score * 0.30 + kw_hit * 0.25)
    return {"relevance": relevance_score, "completeness": completeness_score,
            "keyword_hit": kw_hit, "overall": overall, "mode": "heuristic"}


def _keyword_hit(answer, job_description):
    if not job_description:
        return 50
    kws = extract_keywords(job_description, top_n=15)
    if not kws:
        return 50
    hits = sum(1 for kw in kws if kw in answer.lower())
    return min(int(hits / max(len(kws) * 0.4, 1) * 100), 100)


# ─────────────────────────────────────────────────────────────
# 3. PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────
def score_answer(question: str, answer: str,
                 job_description: str = "",
                 question_type: str = "Behavioral") -> dict:
    if not answer or len(answer.strip()) < 5:
        return {"overall": 0, "mode": "empty"}
    cache = _load_model()
    if cache.get("model_available"):
        return _score_with_model(question, answer, question_type, job_description)
    return _score_heuristic(question, answer, job_description)


def model_is_trained() -> bool:
    return _load_model().get("model_available", False)


def get_model_metrics() -> dict:
    p = ARTEFACT_DIR / "metrics.json"
    if p.exists():
        import json
        return json.load(open(p))
    return {}


# ─────────────────────────────────────────────────────────────
# 4. QUESTION TYPE CLASSIFIER
# ─────────────────────────────────────────────────────────────
_BEHAVIORAL = [
    r"\btell me about\b", r"\bdescribe a (time|situation)\b", r"\bgive (me )?an example\b",
    r"\bhow (did|do) you handle\b", r"\bstrength\b", r"\bweakness\b", r"\bchallenge\b",
    r"\bteamwork\b", r"\bleadership\b", r"\bconflict\b",
]
_TECHNICAL = [
    r"\bexplain\b", r"\bhow does\b", r"\bwhat is\b", r"\bdefine\b", r"\bimplement\b",
    r"\bdesign\b", r"\balgorithm\b", r"\bcode\b", r"\bsystem\b", r"\barchitect\b",
    r"\bperformanc\b", r"\bdatabas\b", r"\bapi\b", r"\bframework\b",
]

def classify_question(question: str) -> str:
    q   = question.lower()
    beh = sum(1 for p in _BEHAVIORAL if re.search(p, q))
    tec = sum(1 for p in _TECHNICAL  if re.search(p, q))
    return "Technical" if tec > beh else "Behavioral"
