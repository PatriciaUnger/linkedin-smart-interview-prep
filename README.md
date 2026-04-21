# LinkedIn Smart Interview Prep

AI-powered interview preparation tool. Paste a LinkedIn job description, get 5 tailored interview questions, and receive scored feedback with concrete coaching tied to a knowledge base of strong and weak example answers.

Built over three iterations for the **Prototyping with Data & AI** course at ESADE (MiBA, 2026). This repo contains the full V3 prototype.

**Live app:** https://linkedin-interview-prep-patricia.streamlit.app/

---

## What This Does

The app walks a candidate through four steps:

1. **Setup** — Paste a job description. The app extracts keywords and generates five role-specific interview questions (mix of behavioural and technical) via Claude.
2. **Practice** — Answer each question in free text. Each answer is scored on five dimensions (structure, specificity, impact, relevance, communication) and paired with coaching feedback that references real examples from a curated knowledge base.
3. **Report** — Competency radar chart, per-question scores, Candidate Mirror (how you come across as a whole), 7-day personalised prep plan. New in V3: a **"Refine this answer"** button on each question that lets you rewrite, re-score, and see a before/after delta.
4. **AI Coach chat** — Multi-turn conversation with full session context injected into the system prompt. Ask anything about your answers and get targeted advice.

## V3 — Evaluation Layer

V3 adds a separate evaluation interface accessible at `?dev=1` in the URL. It is not visible in the normal candidate flow because it is an internal tool for me, not for end users.

The Eval Lab runs any of 10 curated test cases through a 2×2 matrix: {TF-IDF, Semantic} retrieval × {V1-single-prompt, V2-two-step} feedback. It records retrievals, scores, and coaching text for each variant and computes two cross-variant metrics: retrieval overlap and score divergence. Every run is saved as a JSON file under `evaluations/runs/` so results are part of the repository.

**Try it:** https://linkedin-interview-prep-patricia.streamlit.app/?dev=1

## Tech Stack

- **Frontend/UI:** Streamlit
- **LLM:** Anthropic Claude (`claude-sonnet-4-20250514`)
- **Retrieval:** `sentence-transformers` (`all-MiniLM-L6-v2`) with TF-IDF fallback
- **Charts:** Plotly
- **Deployment:** Streamlit Cloud

## Project Structure

```
.
├── app.py                      Main Streamlit app (candidate flow + dev mode router)
├── ai_coach.py                 LLM calls: question generation, scoring, coaching, Mirror, Prep Plan
├── chatbot.py                  Multi-turn AI Coach chatbot
├── nlp_pipeline.py             Keyword extraction + ML fallback scoring
├── rag_engine.py               Retrieval (semantic + TF-IDF modes)
├── interview_kb.py             Curated knowledge base (38 example answers)
├── refine.py                   V3: revise-and-rescore pipeline with delta analysis
├── dev_mode.py                 V3: internal evaluation UI (Eval Lab + Analytics)
├── eval_testset.py             V3: 10 curated test cases for evaluation
├── eval_harness.py             V3: 2×2 variant runner (TF-IDF/Semantic × V1/V2)
├── eval_storage.py             V3: JSON persistence for evaluation runs
├── train_model.py              Trains the optional ML scoring pipeline
├── evaluations/runs/           Saved evaluation runs (JSON, committed to git)
└── artefacts/                  Trained ML pipeline (pickled, optional)
```

## Running Locally

```bash
# 1. Clone
git clone https://github.com/PatriciaUnger/linkedin-smart-interview-prep.git
cd linkedin-smart-interview-prep

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key in Streamlit secrets
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

On first launch the `sentence-transformers` model (~80 MB) will download once and cache. Subsequent runs start in seconds.

## Version History

| Version | Main additions |
|---|---|
| **V1** | Streamlit prototype. Basic question generation, heuristic scoring, single-prompt feedback. |
| **V2** | 4 LLM features added: RAG-augmented feedback with two-step prompt chain, Candidate Mirror (cross-session synthesis), 7-Day Prep Plan, Multi-Turn AI Coach chatbot. Prompt-discipline iteration: 4 rounds to get the Mirror's tone right. |
| **V3** | **Refine this answer** (new user-facing feature with per-dimension before/after deltas). Semantic embeddings replace TF-IDF. Knowledge base expanded from 20 to 38 examples. Internal evaluation layer behind `?dev=1` with 2×2 variant harness, 10 curated test cases, and Analytics dashboard. |

See `2_pager_v3.md` (or the PDF version submitted) for the full V3 write-up.

## Acknowledgements

Built with heavy use of Claude as a coding collaborator. Design decisions (two-step scoring, Refine-as-closing-loop, 2×2 evaluation matrix) were mine; the code generation was Claude's.
