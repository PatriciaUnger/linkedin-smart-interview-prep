"""
rag_engine.py
─────────────
Retrieval engine over example interview answers.

V3 upgrade: supports TWO retrieval modes, selectable at construction time:
    - "tfidf":    original TF-IDF + cosine similarity (v2 baseline)
    - "semantic": sentence-transformers embeddings + cosine similarity (v3)

Both modes share the same public interface, so the app code doesn't need to
know which one is active. The evaluation harness constructs both side-by-side
to compare their retrievals on the same queries.

Why keep TF-IDF?
  - It's the v2 baseline — we need it to measure whether v3 is actually better
  - It's the fallback if sentence-transformers fails to load on Streamlit Cloud
  - It's fast and requires no model download for first-time users

Why MiniLM?
  - 80 MB, runs locally, no API calls, no cost — per the professor's feedback
  - "all-MiniLM-L6-v2" is the canonical small semantic model (~384-dim output)
  - Fits comfortably in Streamlit Cloud's memory budget
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sentence-transformers is imported lazily inside _SemanticRetriever so that
# if the package is missing (e.g. first deploy before requirements.txt is
# updated), the TF-IDF path still works.


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF retriever — identical behaviour to v2, wrapped in a class so both
# retrievers share the same shape.
# ─────────────────────────────────────────────────────────────────────────────
class _TfidfRetriever:
    name = "tfidf"

    def __init__(self, documents):
        self.docs = documents
        texts = [d["text"] for d in documents]

        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
        )
        matrix = self.vec.fit_transform(texts).toarray().astype(np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    def score(self, query: str) -> np.ndarray:
        """Return similarity score per document (same order as self.docs)."""
        q = self.vec.transform([query]).toarray().astype(np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        return self.matrix.dot(q.T).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic retriever — sentence-transformers + cosine similarity.
# ─────────────────────────────────────────────────────────────────────────────
class _SemanticRetriever:
    name = "semantic"
    _model_cache = {}  # class-level cache so we don't reload on every session

    def __init__(self, documents, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.docs = documents
        self.model_name = model_name

        if model_name not in _SemanticRetriever._model_cache:
            _SemanticRetriever._model_cache[model_name] = SentenceTransformer(model_name)
        self.model = _SemanticRetriever._model_cache[model_name]

        texts = [d["text"] for d in documents]
        # encode returns float32 by default; normalize for cosine via dot product
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.matrix = embeddings  # already L2-normalised

    def score(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return self.matrix.dot(q.T).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Public class — same name and interface as v2, so app.py doesn't change.
# ─────────────────────────────────────────────────────────────────────────────
class RAGEngine:
    """
    Retrieval engine over example interview answers.

    Parameters
    ----------
    documents : list of dicts with at least a 'text' and 'quality' key
    mode : "tfidf" | "semantic"
        The retrieval method to use. Defaults to "semantic" (v3). If the
        sentence-transformers package is unavailable, falls back to "tfidf".
    """

    def __init__(self, documents, mode: str = "semantic"):
        self.docs = documents
        self.requested_mode = mode

        if mode == "tfidf":
            self._retriever = _TfidfRetriever(documents)
        elif mode == "semantic":
            try:
                self._retriever = _SemanticRetriever(documents)
            except Exception as e:
                # Graceful fallback — log but keep the app working
                print(f"[RAGEngine] semantic retriever unavailable ({e}); "
                      f"falling back to TF-IDF")
                self._retriever = _TfidfRetriever(documents)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'tfidf' or 'semantic'.")

        self.active_mode = self._retriever.name

    # ── retrieval ──────────────────────────────────────────────────────────
    def _retrieve(self, query, k=3, quality=None):
        scores = self._retriever.score(query)

        indices = list(range(len(self.docs)))
        if quality:
            indices = [i for i in indices if self.docs[i].get("quality") in quality]
        if not indices:
            indices = list(range(len(self.docs)))

        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:k]:
            doc = dict(self.docs[i])
            doc["similarity"] = float(scores[i])
            results.append(doc)
        return results

    def build_rag_context(self, query, k=2):
        """
        Return a formatted string with strong and weak example answers
        similar to the user's query. This gets injected into the LLM prompt.
        """
        strong = self._retrieve(query, k=k, quality=["strong"])
        weak = self._retrieve(query, k=1, quality=["weak"])

        lines = ["## Similar example answers from the knowledge base\n"]

        lines.append("### Strong example(s):")
        for ex in strong:
            lines.append(f"- Skill: {ex.get('skill', '')}")
            lines.append(f"  Answer: \"{ex['text']}\"")
            lines.append(f"  Why it works: {ex.get('why', '')}\n")

        if weak:
            lines.append("### Weak example (what to avoid):")
            ex = weak[0]
            lines.append(f"  Answer: \"{ex['text']}\"")
            lines.append(f"  Why it fails: {ex.get('why', '')}\n")

        return "\n".join(lines)

    # ── eval-harness helpers (new in v3) ───────────────────────────────────
    def retrieve_raw(self, query, k=3, quality=None):
        """
        Returns the raw top-k retrieval results (list of dicts with similarity).
        Used by the evaluation harness to compare retrieval outputs across modes.
        """
        return self._retrieve(query, k=k, quality=quality)
