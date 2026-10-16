"""Phase 2 - Step 2: Data cleaning.

Handles, for both datasets:
  * missing values   (drop unusable rows, impute / flag the rest)
  * duplicates       (exact, then near-duplicates after normalisation)
  * outliers         (too-short / too-long documents, extreme salaries)

Every dropped row is recorded in an audit file so nothing disappears silently.

Outputs
  data/processed/resumes_clean.csv, jobs_clean.csv
  data/processed/resumes_audit.csv,  jobs_audit.csv
  logs/cleaning_report.json
"""
import json

import numpy as np
import pandas as pd

from collect import find_file
from config import (DATASETS, IQR_FENCE_UPPER, LOG_DIR, MAX_JOBS_SAMPLE, MIN_CV_WORDS,
                    MIN_JOB_WORDS, PROCESSED_DIR, RANDOM_STATE, RAW_DIR)
from utils import clean_text, normalize_text, text_hash, word_count


def iqr_upper_fence(series, k=IQR_FENCE_UPPER):
    q1, q3 = series.quantile([0.25, 0.75])
    return q3 + k * (q3 - q1)


# ------------------------------------------------------------------ resumes
def clean_resumes():
    path = find_file(DATASETS["resumes"]["file"])
    if path is None:
        raise SystemExit("Resume.csv not found - run collect.py first.")
    raw = pd.read_csv(path)
    report = {"rows_raw": int(len(raw)),
              "missing_before": raw.isna().sum().to_dict()}

    df = raw.rename(columns={"ID": "resume_id", "Resume_str": "text_raw", "Category": "category"})
    df = df[["resume_id", "text_raw", "category"]].copy()
    df["status"] = "kept"

    # 1) missing values --------------------------------------------------
    df["category"] = df["category"].astype("string").str.strip().str.title()
    df["text_clean"] = df["text_raw"].fillna("").map(clean_text)
    df["word_count"] = df["text_clean"].map(word_count)
    no_text = (df["text_clean"].eq("") | df["category"].isna() | df["category"].eq("").fillna(False)).astype(bool)
    df.loc[no_text, "status"] = "dropped_missing"

    # 2) duplicates ------------------------------------------------------
    live = df["status"].eq("kept")
    dup_exact = live & df.duplicated("text_clean", keep="first")
    df.loc[dup_exact, "status"] = "dropped_duplicate_exact"

    df["norm_hash"] = df["text_clean"].map(lambda t: text_hash(normalize_text(t)))
    live = df["status"].eq("kept")
    dup_near = live & df.duplicated("norm_hash", keep="first")
    df.loc[dup_near, "status"] = "dropped_duplicate_near"

    # 3) outliers --------------------------------------------------------
    live = df["status"].eq("kept")
    upper = iqr_upper_fence(df.loc[live, "word_count"])
    df.loc[live & (df["word_count"] < MIN_CV_WORDS), "status"] = "dropped_outlier_short"
    df.loc[live & (df["word_count"] > upper), "status"] = "dropped_outlier_long"

    clean = df[df["status"].eq("kept")].drop(columns=["status", "norm_hash", "text_raw"])
    clean.to_csv(PROCESSED_DIR / "resumes_clean.csv", index=False)
    df[["resume_id", "category", "word_count", "status"]].to_csv(PROCESSED_DIR / "resumes_audit.csv", index=False)

    report.update({
        "rows_clean": int(len(clean)),
        "dropped": df["status"].value_counts().drop("kept", errors="ignore").to_dict(),
        "outlier_rules": {"min_words": MIN_CV_WORDS, "max_words_iqr_fence": round(float(upper), 1),
                          "fence_k": IQR_FENCE_UPPER},
        "classes": int(clean["category"].nunique()),
    })
    print(f"[resumes] {report['rows_raw']:,} -> {report['rows_clean']:,} rows | dropped: {report['dropped']}")
    return report


# --------------------------------------------------------------------- jobs
JOB_COLS = ["job_id", "company_name", "title", "description", "skills_desc", "location",
            "formatted_work_type", "formatted_experience_level", "remote_allowed",
            "max_salary", "min_salary", "pay_period", "views", "applies"]
PERIOD_TO_YEAR = {"HOURLY": 2080, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12, "YEARLY": 1}


def clean_jobs():
    path = find_file(DATASETS["jobs"]["file"])
    if path is None:
        raise SystemExit("postings.csv not found - run collect.py first.")
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in JOB_COLS if c in header]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in JOB_COLS:                      # tolerate columns missing in other dataset versions
        if col not in raw:
            raw[col] = np.nan

    report = {"rows_raw": int(len(raw)), "missing_before": raw.isna().sum().to_dict()}
    if len(raw) > MAX_JOBS_SAMPLE:
        raw = raw.sample(MAX_JOBS_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
        report["sampled_to"] = MAX_JOBS_SAMPLE

    df = raw.copy()
    df["status"] = "kept"

    # 1) missing values --------------------------------------------------
    df["title"] = df["title"].fillna("").map(clean_text)
    df["description"] = df["description"].fillna("").map(clean_text)
    df["skills_desc"] = df["skills_desc"].fillna("").map(clean_text)
    df["has_skills_desc"] = df["skills_desc"].ne("").astype(int)
    df["word_count"] = df["description"].map(word_count)
    df.loc[df["title"].eq("") | df["description"].eq(""), "status"] = "dropped_missing"

    for col in ("formatted_work_type", "formatted_experience_level", "company_name", "location"):
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df["remote_allowed"] = pd.to_numeric(df["remote_allowed"], errors="coerce").fillna(0).astype(int)

    # 2) duplicates ------------------------------------------------------
    live = df["status"].eq("kept")
    df.loc[live & df.duplicated("job_id", keep="first") & df["job_id"].notna(), "status"] = "dropped_duplicate_id"
    key = (df["title"].str.lower() + "|" + df["company_name"].str.lower() + "|"
           + df["description"].map(lambda t: text_hash(normalize_text(t))))
    live = df["status"].eq("kept")
    df.loc[live & key.duplicated(keep="first"), "status"] = "dropped_duplicate_repost"

    # 3) outliers --------------------------------------------------------
    live = df["status"].eq("kept")
    upper = iqr_upper_fence(df.loc[live, "word_count"])
    df.loc[live & (df["word_count"] < MIN_JOB_WORDS), "status"] = "dropped_outlier_short"
    df.loc[live & (df["word_count"] > upper), "status"] = "dropped_outlier_long"

    # salary: convert to yearly, drop impossible values, winsorise the extremes (1st-99th pct)
    factor = df["pay_period"].astype(str).str.upper().map(PERIOD_TO_YEAR)
    mid = df[["min_salary", "max_salary"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    salary = mid * factor
    salary[(salary <= 0) | (salary > 5_000_000)] = np.nan
    lo, hi = salary.quantile([0.01, 0.99])
    df["salary_yearly"] = salary.clip(lo, hi)

    # numeric engagement columns: median-impute + flag
    for col in ("views", "applies"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    clean = df[df["status"].eq("kept")].drop(columns=["status", "min_salary", "max_salary", "pay_period"])
    clean.to_csv(PROCESSED_DIR / "jobs_clean.csv", index=False)
    df[["job_id", "word_count", "status"]].to_csv(PROCESSED_DIR / "jobs_audit.csv", index=False)

    report.update({
        "rows_clean": int(len(clean)),
        "dropped": df["status"].value_counts().drop("kept", errors="ignore").to_dict(),
        "outlier_rules": {"min_words": MIN_JOB_WORDS, "max_words_iqr_fence": round(float(upper), 1),
                          "salary_winsorised_to": [round(float(lo), 0), round(float(hi), 0)]},
        "salary_missing_share": round(float(clean["salary_yearly"].isna().mean()), 3),
    })
    print(f"[jobs]    {report['rows_raw']:,} -> {report['rows_clean']:,} rows | dropped: {report['dropped']}")
    return report


def main():
    report = {"resumes": clean_resumes(), "jobs": clean_jobs()}
    (LOG_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Cleaning report written to {LOG_DIR / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
