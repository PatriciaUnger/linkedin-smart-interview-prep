"""
train_model.py
══════════════
OFFLINE training script – run this ONCE to produce artefacts/model.pkl
The Streamlit app (app.py) only loads the saved model, never retrains.

Pipeline
────────
1. Generate / load dataset
2. Data cleaning   (missing values, outliers, type fixes)
3. Feature engineering  (text features → numeric)
4. sklearn ColumnTransformer  (Imputer + Encoder + Scaler)
5. Train / test split  (80 / 20, stratified)
6. Cross-validation  (5-fold StratifiedKFold)
7. Grid search over two candidate models
8. Evaluate best model on held-out test set
9. Save artefacts (model, pipeline, label-encoder, metrics)
"""

import os, warnings, json
import numpy as np
import pandas as pd

from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing      import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics            import (classification_report, accuracy_score,
                                        f1_score, confusion_matrix)
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# 1.  GENERATE DATASET
# ══════════════════════════════════════════════════════════════
"""
Dataset description
───────────────────
Each row represents ONE interview answer attempt with the following columns:

  answer_text        : raw answer text (string)
  question_type      : "Behavioral" or "Technical"
  word_count         : number of words in the answer
  has_example        : 1 if answer contains "for example / e.g. / instance"
  has_numbers        : 1 if answer contains any digit
  uses_first_person  : 1 if "I " appears in answer
  answer_length_cat  : "short" / "medium" / "long"  (categorical)
  avg_sentence_len   : average words per sentence
  quality_label      : target → "weak" / "good" / "strong"

Missing values (~5 %) and noisy rows are injected to make cleaning realistic.
"""

WEAK_ANSWERS = [
    "I don't know much about that.",
    "I would try my best.",
    "Yes I have experience with that.",
    "I think I am good at it.",
    "I worked on several projects.",
    "That's a good question.",
    "I always do my best work.",
    "I can handle pressure fine.",
    "I have done similar things before.",
    "Not sure but I would figure it out.",
    "I managed a team once.",
    "I usually work hard.",
    "I am a fast learner.",
    "I have seen this kind of problem.",
    "I would ask my manager for help.",
]

GOOD_ANSWERS = [
    "In my previous role I led a cross-functional team of five engineers. We faced a tight deadline and I organised daily stand-ups to track blockers. We delivered on time and reduced bug count by 20 percent.",
    "I use Python and SQL daily. For example, I built an ETL pipeline that processed 10 million rows per day using pandas and SQLAlchemy, cutting report generation time from 4 hours to 15 minutes.",
    "When I encounter conflict in a team I first try to understand the other perspective. In one case a colleague and I disagreed on architecture. I proposed we prototype both approaches over two days and measure performance, which resolved the disagreement objectively.",
    "I prioritise tasks using the Eisenhower matrix. Last quarter I managed three parallel projects by breaking each into weekly milestones, communicating progress to stakeholders every Friday, and escalating blockers immediately.",
    "I define success by measurable outcomes. In my last project the KPI was reducing customer churn by 10 percent. I built a logistic regression model on 6 months of user behaviour data, achieving 87 percent AUC, which directly contributed to a 12 percent churn reduction.",
    "My greatest strength is structured problem-solving. For example I once inherited a codebase with no documentation. I spent the first week mapping dependencies, wrote a diagram, then onboarded two new engineers using it within a month.",
    "I keep up with industry trends by reading papers on arXiv weekly, following key researchers on Twitter, and implementing one new technique per quarter in a personal project.",
    "To handle ambiguous requirements I schedule a clarification meeting within 24 hours, document assumptions in writing, and confirm them with the stakeholder before starting work.",
]

STRONG_ANSWERS = [
    "Situation: Our recommendation system had a latency of 800ms which was causing a 15 percent drop-off. Task: I was asked to reduce it to under 200ms. Action: I profiled the service, identified that 70 percent of latency came from a redundant database call, cached results with Redis, and vectorised the scoring loop using NumPy. Result: Latency dropped to 140ms, drop-off decreased by 12 percent, and we saved roughly 30 percent in compute costs.",
    "Using the STAR method: in my last role our data pipeline was failing silently every weekend. I designed an alerting system using Airflow sensors and Slack webhooks. I also added a data quality layer using Great Expectations. This caught 98 percent of anomalies before they reached production dashboards and reduced incident response time from 4 hours to 20 minutes.",
    "I approached this problem by first conducting a root cause analysis using the 5-Whys technique. I discovered the underlying issue was inconsistent data encoding across three upstream systems. I standardised encoding in a shared utility library, wrote unit tests for edge cases, and documented the decision in our ADR system. The fix eliminated that class of bug entirely for the following 18 months.",
    "To scale our ML training pipeline I implemented a distributed training strategy using PyTorch DDP across 4 GPUs. I also introduced mixed-precision training and gradient checkpointing which reduced GPU memory by 40 percent and cut training time from 14 hours to 3.5 hours while maintaining model accuracy within 0.2 percent of the baseline.",
    "My approach to technical debt is to treat it as a first-class backlog item. I introduced a rule that 20 percent of every sprint is reserved for tech debt. I built a dashboard tracking cyclomatic complexity over time and presented it to leadership as a risk metric. Over 6 months we reduced our critical debt score by 60 percent while maintaining feature velocity.",
]

def generate_dataset(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []

    for _ in range(n):
        label = rng.choice(["weak", "good", "strong"], p=[0.35, 0.40, 0.25])
        q_type = rng.choice(["Behavioral", "Technical"], p=[0.55, 0.45])

        if label == "weak":
            base = rng.choice(WEAK_ANSWERS)
            # add small noise
            base = base + " " + rng.choice(["I think.", "Maybe.", "Not sure.", ""])
        elif label == "good":
            base = rng.choice(GOOD_ANSWERS)
        else:
            base = rng.choice(STRONG_ANSWERS)

        words      = base.split()
        word_count = len(words)
        sentences  = [s.strip() for s in base.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        avg_sent   = word_count / max(len(sentences), 1)

        rows.append({
            "answer_text":       base,
            "question_type":     q_type,
            "word_count":        word_count,
            "has_example":       int(any(kw in base.lower() for kw in ["example", "e.g.", "instance", "for instance"])),
            "has_numbers":       int(any(c.isdigit() for c in base)),
            "uses_first_person": int(" i " in base.lower() or base.lower().startswith("i ")),
            "avg_sentence_len":  round(avg_sent, 2),
            "answer_length_cat": "short" if word_count < 30 else ("long" if word_count > 100 else "medium"),
            "quality_label":     label,
        })

    df = pd.DataFrame(rows)

    # ── inject realistic missing values (~5 %) ──────────────
    for col in ["word_count", "avg_sentence_len", "has_numbers", "question_type"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    # ── inject some nonsense / outlier rows ─────────────────
    outlier_idx = rng.choice(n, size=20, replace=False)
    df.loc[outlier_idx, "word_count"] = rng.choice([-1, 0, 9999], size=20)

    return df


# ══════════════════════════════════════════════════════════════
# 2.  DATA CLEANING
# ══════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 2: Data Cleaning ─────────────────────────────")
    df = df.copy()

    initial_rows = len(df)
    print(f"   Rows before cleaning : {initial_rows}")

    # 2a. Drop rows with no answer text or no label
    df = df.dropna(subset=["answer_text", "quality_label"])
    print(f"   After dropping null target/text : {len(df)}")

    # 2b. Strip whitespace from strings
    df["answer_text"]   = df["answer_text"].str.strip()
    df["question_type"] = df["question_type"].astype(str).str.strip()

    # 2c. Remove rows where answer_text is empty or too short (< 3 words)
    df = df[df["answer_text"].str.split().str.len() >= 3]
    print(f"   After removing too-short answers : {len(df)}")

    # 2d. Fix outlier word_count values (negative or impossibly large)
    Q1 = df["word_count"].quantile(0.01)
    Q3 = df["word_count"].quantile(0.99)
    df.loc[(df["word_count"] < 0) | (df["word_count"] > Q3 * 3), "word_count"] = np.nan
    print(f"   Outlier word_count values replaced with NaN")

    # 2e. Fill remaining missing numeric values with column median
    for col in ["word_count", "avg_sentence_len", "has_numbers",
                "has_example", "uses_first_person"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # 2f. Fill missing categorical with mode
    for col in ["question_type", "answer_length_cat"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   Imputed {n_missing} missing values in '{col}' with mode='{mode_val}'")

    # 2g. Normalise question_type (handle "nan" strings from fillna)
    df["question_type"] = df["question_type"].replace("nan", "Behavioral")
    df["question_type"] = df["question_type"].where(
        df["question_type"].isin(["Behavioral", "Technical"]), "Behavioral"
    )

    # 2h. Cast types
    df["word_count"]        = df["word_count"].astype(float)
    df["avg_sentence_len"]  = df["avg_sentence_len"].astype(float)
    df["has_example"]       = df["has_example"].astype(int)
    df["has_numbers"]       = df["has_numbers"].astype(int)
    df["uses_first_person"] = df["uses_first_person"].astype(int)

    print(f"   Rows after cleaning : {len(df)}  (removed {initial_rows - len(df)})")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 3: Feature Engineering ──────────────────────")
    df = df.copy()

    # Recalculate word_count from text (more reliable than stored value)
    df["word_count"] = df["answer_text"].str.split().str.len().astype(float)

    # Lexical richness: unique words / total words
    df["lexical_richness"] = df["answer_text"].apply(
        lambda t: len(set(t.lower().split())) / max(len(t.split()), 1)
    )

    # Contains STAR keywords (behavioural signal)
    star_kw = ["situation", "task", "action", "result", "achieved", "outcome",
               "reduced", "increased", "improved", "delivered", "percent", "%"]
    df["star_score"] = df["answer_text"].apply(
        lambda t: sum(1 for kw in star_kw if kw in t.lower())
    ).astype(float)

    # Sentence count
    df["sentence_count"] = df["answer_text"].apply(
        lambda t: max(len([s for s in t.replace("!", ".").replace("?", ".").split(".") if s.strip()]), 1)
    ).astype(float)

    # Answer is very short flag
    df["is_very_short"] = (df["word_count"] < 20).astype(int)

    print(f"   New features added: lexical_richness, star_score, sentence_count, is_very_short")
    return df


# ══════════════════════════════════════════════════════════════
# 4.  BUILD SKLEARN PIPELINE
# ══════════════════════════════════════════════════════════════
"""
We use a ColumnTransformer to handle three types of features:

  A) Numeric features  → SimpleImputer(median) + StandardScaler
  B) Categorical features → SimpleImputer(most_frequent) + OneHotEncoder
  C) Text feature (answer_text) → TfidfVectorizer (max 300 features)

The ColumnTransformer output is fed into a classifier.
"""

NUMERIC_FEATURES = [
    "word_count", "avg_sentence_len", "has_example", "has_numbers",
    "uses_first_person", "lexical_richness", "star_score",
    "sentence_count", "is_very_short",
]
CATEGORICAL_FEATURES = ["question_type", "answer_length_cat"]
TEXT_FEATURE = "answer_text"

def build_pipeline(classifier) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    text_transformer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num",  numeric_transformer,     NUMERIC_FEATURES),
        ("cat",  categorical_transformer, CATEGORICAL_FEATURES),
        ("text", text_transformer,        TEXT_FEATURE),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ══════════════════════════════════════════════════════════════
# 5 & 6.  TRAIN / TEST SPLIT + CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame):
    print("\n── Step 5: Train / Test Split ────────────────────────")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
    y_raw = df["quality_label"]

    # Encode labels: weak=0, good=1, strong=2
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"   Classes: {le.classes_}  →  {list(range(len(le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train set : {len(X_train)} rows")
    print(f"   Test set  : {len(X_test)} rows")

    # ── Candidate models (manual grid search) ───────────────
    print("\n── Step 6: 5-Fold Cross-Validation ──────────────────")

    candidates = {
        "LogisticRegression (C=0.1)": build_pipeline(
            LogisticRegression(C=0.1, max_iter=500, random_state=42)
        ),
        "LogisticRegression (C=1.0)": build_pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=42)
        ),
        "RandomForest (n=100)": build_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "RandomForest (n=200)": build_pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1_weighted"],
            return_train_score=True,
        )
        mean_val_acc = scores["test_accuracy"].mean()
        mean_val_f1  = scores["test_f1_weighted"].mean()
        cv_results[name] = {
            "mean_val_accuracy": round(mean_val_acc, 4),
            "mean_val_f1":       round(mean_val_f1,  4),
            "std_val_accuracy":  round(scores["test_accuracy"].std(), 4),
        }
        print(f"   {name:<35}  val_acc={mean_val_acc:.3f}  val_f1={mean_val_f1:.3f}")

    # ── Select best model by validation F1 ──────────────────
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_val_f1"])
    print(f"\n   ✓ Best model: {best_name}")

    best_pipeline = candidates[best_name]
    best_pipeline.fit(X_train, y_train)

    # ── Final evaluation on held-out test set ───────────────
    print("\n── Step 7: Test Set Evaluation ───────────────────────")
    y_pred   = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"   Test Accuracy : {test_acc:.3f}")
    print(f"   Test F1 (weighted) : {test_f1:.3f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    metrics = {
        "best_model":        best_name,
        "cv_results":        cv_results,
        "test_accuracy":     round(test_acc, 4),
        "test_f1_weighted":  round(test_f1,  4),
        "classes":           list(le.classes_),
        "feature_cols":      NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE],
    }

    return best_pipeline, le, metrics


# ══════════════════════════════════════════════════════════════
# 8.  SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════
def save_artefacts(pipeline, le, metrics):
    os.makedirs("artefacts", exist_ok=True)
    joblib.dump(pipeline, "artefacts/model_pipeline.pkl")
    joblib.dump(le,       "artefacts/label_encoder.pkl")
    with open("artefacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n── Artefacts Saved ────────────────────────────────────")
    print("   artefacts/model_pipeline.pkl")
    print("   artefacts/label_encoder.pkl")
    print("   artefacts/metrics.json")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("══════════════════════════════════════════════════════")
    print("  InterviewIQ — Model Training Script")
    print("══════════════════════════════════════════════════════")

    print("\n── Step 1: Generate Dataset ──────────────────────────")
    df_raw = generate_dataset(n=2000)
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/interview_answers.csv", index=False)
    print(f"   Generated {len(df_raw)} rows → saved to data/interview_answers.csv")
    print(f"   Columns: {list(df_raw.columns)}")
    print(f"   Missing values per column:\n{df_raw.isna().sum().to_string()}")

    df_clean = clean_data(df_raw)
    df_feat  = engineer_features(df_clean)

    pipeline, le, metrics = train_and_evaluate(df_feat)
    save_artefacts(pipeline, le, metrics)

    print("\n✅  Training complete!\n")
