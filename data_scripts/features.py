"""Phase 2 - Step 3: Feature strategy (normalisation, scaling, encoding).

CV features (hand-crafted, explainable - these feed the transparent quality score):
    length, sentence stats, lexical diversity, quantified impact (numbers that are not years),
    action-verb density, section coverage, skill coverage.
Text normalisation + TF-IDF for the CV text.
Scaling:  skewed counts -> log1p, then StandardScaler fitted on the TRAIN split only (no leakage).
Encoding: category -> integer label (LabelEncoder); job work-type / experience level -> one-hot.

Outputs
  data/processed/resumes_features.csv, jobs_features.csv
  data/processed/artifacts/*.joblib, resumes_tfidf_*.npz
  logs/feature_report.json, cv_skill_share.csv, job_skill_demand.csv, top_terms_per_category.json
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (ACTION_VERBS, LOG_DIR, PROCESSED_DIR, RANDOM_STATE, SECTION_KEYWORDS, SKILLS)
from utils import extract_skills, normalize_text

ART_DIR = PROCESSED_DIR / "artifacts"
ART_DIR.mkdir(exist_ok=True)

_YEAR_RE = re.compile(r"^(19|20)\d\d$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ------------------------------------------------------------ CV feature maths
def cv_features(text, norm):
    tokens = norm.split()
    n = max(len(tokens), 1)
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()] or [text]
    digit_tokens = re.findall(r"\d[\d,\.]*", text)
    metric_numbers = [d for d in digit_tokens if not _YEAR_RE.match(d.strip(",."))]   # years are not impact
    quantified_sents = sum(1 for s in sentences if any(not _YEAR_RE.match(d.strip(",."))
                                                        for d in re.findall(r"\d[\d,\.]*", s)))
    verbs = sum(1 for t in tokens if t in ACTION_VERBS)
    sections = sum(any(kw in norm for kw in kws) for kws in SECTION_KEYWORDS.values())
    first = tokens[:300]                                                     # length-robust diversity
    return {
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "avg_sentence_len": len(tokens) / len(sentences),
        "avg_word_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "lexical_diversity": len(set(first)) / max(len(first), 1),
        "metric_number_count": len(metric_numbers),
        "percent_count": text.count("%"),
        "numbers_per_100_words": 100 * len(metric_numbers) / n,
        "quantified_sentence_ratio": quantified_sents / len(sentences),
        "action_verb_count": verbs,
        "action_verbs_per_100_words": 100 * verbs / n,
        "section_count": sections,
    }


def scale_block(train_df, all_df, cols, name, report):
    """log1p on strongly skewed columns, then z-score using TRAIN statistics only."""
    work = all_df[cols].astype(float).copy()
    skewed = [c for c in cols if train_df[c].astype(float).skew() > 1.0 and (train_df[c] >= 0).all()]
    for c in skewed:
        work[c] = np.log1p(work[c])
    scaler = StandardScaler().fit(work.loc[train_df.index])
    scaled = pd.DataFrame(scaler.transform(work), columns=[f"{c}_z" for c in cols], index=all_df.index)
    joblib.dump({"scaler": scaler, "columns": cols, "log1p_columns": skewed}, ART_DIR / f"{name}_scaler.joblib")
    report[f"{name}_log1p_columns"] = skewed
    return scaled


# ------------------------------------------------------------------- resumes
def build_resume_features(report):
    df = pd.read_csv(PROCESSED_DIR / "resumes_clean.csv")
    df["text_norm"] = df["text_clean"].map(normalize_text)

    feats = pd.DataFrame([cv_features(t, n) for t, n in zip(df["text_clean"], df["text_norm"])])
    skills_found = df["text_norm"].map(extract_skills)
    feats["skill_count"] = skills_found.map(len)
    df = pd.concat([df.drop(columns=["word_count"]), feats], axis=1)   # recomputed on normalised tokens

    # encoding: category -> integer id
    le = LabelEncoder().fit(df["category"])
    df["category_id"] = le.transform(df["category"])
    joblib.dump(le, ART_DIR / "category_label_encoder.joblib")

    # stratified train/test split (scaler + TF-IDF are fitted on train only)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["category_id"],
                                           random_state=RANDOM_STATE)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    train = df.loc[train_idx]

    num_cols = list(feats.columns)
    scaled = scale_block(train, df, num_cols, "resume", report)
    out = pd.concat([df[["resume_id", "category", "category_id", "split"] + num_cols], scaled], axis=1)
    out.to_csv(PROCESSED_DIR / "resumes_features.csv", index=False)

    # text normalisation -> TF-IDF (fit on train only)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english",
                            min_df=3, sublinear_tf=True)
    x_train = tfidf.fit_transform(df.loc[train_idx, "text_norm"])
    x_test = tfidf.transform(df.loc[test_idx, "text_norm"])
    sparse.save_npz(PROCESSED_DIR / "resumes_tfidf_train.npz", x_train)
    sparse.save_npz(PROCESSED_DIR / "resumes_tfidf_test.npz", x_test)
    joblib.dump(tfidf, ART_DIR / "resume_tfidf.joblib")

    terms = np.array(tfidf.get_feature_names_out())
    top = {}
    y_train = df.loc[train_idx, "category"].to_numpy()
    for cat in le.classes_:
        rows = x_train[y_train == cat]
        if rows.shape[0]:
            mean = np.asarray(rows.mean(axis=0)).ravel()
            top[cat] = terms[mean.argsort()[::-1][:10]].tolist()
    (LOG_DIR / "top_terms_per_category.json").write_text(json.dumps(top, indent=2))

    # skill coverage in CVs (used for the skill-gap preview chart)
    n = len(df)
    counts = pd.Series([s for found in skills_found for s in found]).value_counts()
    cv_share = (counts / n).reindex(SKILLS).fillna(0).rename("cv_share")
    cv_share.to_csv(LOG_DIR / "cv_skill_share.csv", index_label="skill")

    report["resumes"] = {
        "rows": int(n), "classes": int(len(le.classes_)),
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "numeric_features": num_cols, "tfidf_vocab_size": int(len(terms)),
        "label_mapping": {c: int(i) for i, c in enumerate(le.classes_)},
    }
    print(f"[resumes] {n:,} rows, {len(num_cols)} numeric features, TF-IDF vocab {len(terms)}")


# ---------------------------------------------------------------------- jobs
def build_job_features(report):
    df = pd.read_csv(PROCESSED_DIR / "jobs_clean.csv", low_memory=False)
    text = (df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["skills_desc"].fillna(""))
    norm = text.map(normalize_text)
    skills_found = norm.map(extract_skills)

    df["title_word_count"] = df["title"].str.split().str.len()
    df["skills_desc_len"] = df["skills_desc"].fillna("").str.split().str.len()
    df["skill_count"] = skills_found.map(len)
    df["salary_missing"] = df["salary_yearly"].isna().astype(int)
    df["salary_yearly"] = df["salary_yearly"].fillna(df["salary_yearly"].median())

    train_idx, _ = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE)
    num_cols = ["word_count", "title_word_count", "skills_desc_len", "skill_count",
                "salary_yearly", "views", "applies"]
    scaled = scale_block(df.loc[train_idx], df, num_cols, "job", report)

    # one-hot encoding of the categoricals
    onehot = pd.get_dummies(df[["formatted_work_type", "formatted_experience_level"]],
                            prefix=["work_type", "exp_level"], dtype=int)
    onehot.columns = [re.sub(r"\W+", "_", c).strip("_").lower() for c in onehot.columns]
    flags = df[["remote_allowed", "has_skills_desc", "salary_missing"]]

    out = pd.concat([df[["job_id", "title", "formatted_work_type", "formatted_experience_level"]],
                     df[num_cols], scaled, onehot, flags], axis=1)
    out.to_csv(PROCESSED_DIR / "jobs_features.csv", index=False)
    joblib.dump({"onehot_columns": list(onehot.columns)}, ART_DIR / "job_onehot_columns.joblib")

    # skill demand across postings (used for the skill-gap preview chart)
    n = len(df)
    counts = pd.Series([s for found in skills_found for s in found]).value_counts()
    demand = (counts / n).reindex(SKILLS).fillna(0).rename("job_share")
    demand.to_csv(LOG_DIR / "job_skill_demand.csv", index_label="skill")

    report["jobs"] = {"rows": int(n), "numeric_features": num_cols,
                      "onehot_columns": list(onehot.columns)}
    print(f"[jobs]    {n:,} rows, {len(num_cols)} numeric + {onehot.shape[1]} one-hot features")


def main():
    report = {}
    build_resume_features(report)
    build_job_features(report)
    (LOG_DIR / "feature_report.json").write_text(json.dumps(report, indent=2))
    print(f"Feature report written to {LOG_DIR / 'feature_report.json'}")


if __name__ == "__main__":
    main()
