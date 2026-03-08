import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class RAGEngine:
    """
    Simple retrieval engine over a list of example answers.
    Uses TF-IDF vectors + cosine similarity to find the most relevant examples
    for a given user answer, then builds a context string for the LLM prompt.
    """

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

        # normalise rows so dot product gives cosine similarity directly
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    def _retrieve(self, query, k=3, quality=None):
        q = self.vec.transform([query]).toarray().astype(np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n

        scores = self.matrix.dot(q.T).flatten()

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
