"""Phase 2 - Step 4: Exploratory data analysis (charts for the LaTeX report).

Reads the outputs of clean.py / features.py and writes PNG figures to figures/.
Run after features.py.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG_DIR, LOG_DIR, PROCESSED_DIR

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
BLUE, ORANGE, GREY = "#2F5496", "#E07B39", "#9AA3AE"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def fig_cv_class_distribution(cv):
    counts = cv["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(counts.index, counts.values, color=BLUE)
    for y, v in enumerate(counts.values):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=8)
    ax.set_xlabel("Number of CVs")
    ax.set_title("CV dataset: class distribution (after cleaning)")
    save(fig, "fig01_cv_category_distribution.png")


def fig_cv_length_before_after(audit):
    kept = audit[audit["status"] == "kept"]["word_count"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, max(audit["word_count"].quantile(0.995), 1), 60)
    axes[0].hist(audit["word_count"].clip(upper=bins[-1]), bins=bins, color=GREY, label="all rows (raw)")
    axes[0].hist(kept.clip(upper=bins[-1]), bins=bins, color=BLUE, alpha=0.85, label="kept after cleaning")
    axes[0].set_xlabel("Words per CV")
    axes[0].set_ylabel("CVs")
    axes[0].legend()
    axes[0].set_title("CV length distribution")
    axes[1].boxplot([audit["word_count"], kept], showfliers=True,
                    flierprops={"markersize": 3, "markerfacecolor": ORANGE, "markeredgecolor": ORANGE})
    axes[1].set_xticks([1, 2], ["raw", "cleaned"])
    axes[1].set_ylabel("Words per CV")
    axes[1].set_title("Outlier handling")
    save(fig, "fig02_cv_length_before_after.png")


def fig_cv_correlation(feat):
    cols = [c for c in feat.columns if not c.endswith("_z") and c not in ("resume_id", "category", "category_id", "split")]
    corr = feat[cols].corr().fillna(0)   # constant columns have undefined correlation
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("CV engineered features: correlation matrix")
    save(fig, "fig03_cv_feature_correlation.png")


def fig_cv_quantification(feat):
    g = feat.groupby("category")["numbers_per_100_words"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(g.index, g.values, color=ORANGE)
    ax.set_xlabel("Average numbers (non-year) per 100 words")
    ax.set_title("How quantified are CVs, by category?")
    save(fig, "fig04_cv_quantification_by_category.png")


def fig_missing_values(report):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(axes, [("resumes", "CV dataset"), ("jobs", "Job postings (selected columns)")]):
        miss = pd.Series(report[key]["missing_before"])
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            ax.text(0.5, 0.5, "no missing values", ha="center", va="center")
            ax.axis("off")
        else:
            ax.barh(miss.index, miss.values, color=BLUE)
            ax.set_xlabel("Missing cells (raw data)")
        ax.set_title(title)
    save(fig, "fig05_missing_values.png")


def fig_job_categoricals(jobs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["formatted_experience_level", "formatted_work_type"],
                              ["Experience level", "Work type"]):
        vc = jobs[col].value_counts().sort_values()
        ax.barh(vc.index, vc.values, color=BLUE)
        ax.set_title(f"Job postings: {title}")
        ax.set_xlabel("Postings")
    save(fig, "fig06_job_categorical_distributions.png")


def fig_job_length(jobs, audit):
    fig, ax = plt.subplots(figsize=(7, 4))
    cap = audit["word_count"].quantile(0.995)
    ax.hist(audit["word_count"].clip(upper=cap), bins=60, color=GREY, label="raw sample")
    ax.hist(jobs["word_count"].clip(upper=cap), bins=60, color=BLUE, alpha=0.85, label="cleaned")
    ax.set_xlabel("Words per job description")
    ax.set_ylabel("Postings")
    ax.legend()
    ax.set_title("Job description length")
    save(fig, "fig07_job_description_length.png")


def fig_skill_gap_preview():
    jobs = pd.read_csv(LOG_DIR / "job_skill_demand.csv", index_col="skill")["job_share"]
    cvs = pd.read_csv(LOG_DIR / "cv_skill_share.csv", index_col="skill")["cv_share"]
    top = jobs.sort_values(ascending=False).head(15).index[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + 0.2, jobs[top] * 100, height=0.4, color=BLUE, label="% of job postings requiring it")
    ax.barh(y - 0.2, cvs[top] * 100, height=0.4, color=ORANGE, label="% of CVs mentioning it")
    ax.set_yticks(y, top)
    ax.set_xlabel("Share (%)")
    ax.set_title("Skill demand vs. skill supply (preview of CVision's skill-gap chart)")
    ax.legend(loc="lower right")
    save(fig, "fig08_skill_gap_preview.png")


def fig_scaling_effect(raw_feat):
    col = "metric_number_count"
    raw = raw_feat[col].astype(float)
    z = raw_feat[f"{col}_z"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(raw, bins=50, color=GREY)
    axes[0].set_title(f"{col}: raw (skew = {raw.skew():.2f})")
    axes[1].hist(z, bins=50, color=BLUE)
    axes[1].set_title(f"{col}: log1p + z-score (skew = {z.skew():.2f})")
    for ax in axes:
        ax.set_ylabel("CVs")
    save(fig, "fig09_scaling_effect.png")


def main():
    cv = pd.read_csv(PROCESSED_DIR / "resumes_clean.csv")
    cv_audit = pd.read_csv(PROCESSED_DIR / "resumes_audit.csv")
    cv_feat = pd.read_csv(PROCESSED_DIR / "resumes_features.csv")
    jobs = pd.read_csv(PROCESSED_DIR / "jobs_clean.csv", low_memory=False)
    job_audit = pd.read_csv(PROCESSED_DIR / "jobs_audit.csv")
    report = json.loads((LOG_DIR / "cleaning_report.json").read_text())

    print("Generating figures ...")
    fig_cv_class_distribution(cv)
    fig_cv_length_before_after(cv_audit)
    fig_cv_correlation(cv_feat)
    fig_cv_quantification(cv_feat)
    fig_missing_values(report)
    fig_job_categoricals(jobs)
    fig_job_length(jobs, job_audit)
    fig_skill_gap_preview()
    fig_scaling_effect(cv_feat)

    summary = {
        "cv_rows": int(len(cv)),
        "cv_class_imbalance_ratio": round(float(cv["category"].value_counts().max() / cv["category"].value_counts().min()), 2),
        "cv_median_words": float(cv["word_count"].median()),
        "job_rows": int(len(jobs)),
        "job_median_words": float(jobs["word_count"].median()),
    }
    (LOG_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("EDA summary:", summary)


if __name__ == "__main__":
    main()
