"""Central configuration for all Phase 2 scripts (paths, constants, vocabularies)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CVISION_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path(os.environ.get("CVISION_FIG_DIR", ROOT / "figures"))
LOG_DIR = Path(os.environ.get("CVISION_LOG_DIR", ROOT / "logs"))

for _d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------- datasets
DATASETS = {
    "resumes": {
        "kaggle_slug": "snehaanbhawal/resume-dataset",
        "url": "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset",
        "file": "Resume.csv",
        "kaggle_file": None,  # download the whole (small) dataset
        "purpose": "CV text + job category; used for CV-quality features and section/writing analysis.",
    },
    "jobs": {
        "kaggle_slug": "arshkon/linkedin-job-postings",
        "url": "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings",
        "file": "postings.csv",
        "kaggle_file": "postings.csv",  # only the postings table is needed
        "purpose": "Job ads; used for role-fit scoring and the skill-gap chart.",
    },
}

# ---------------------------------------------------------------- cleaning
MIN_CV_WORDS = 50          # shorter than this is not a real CV
MIN_JOB_WORDS = 30
IQR_FENCE_UPPER = 3.0      # far-outlier fence for very long documents
MAX_JOBS_SAMPLE = 40000    # random sample of the 124k postings (keeps runs fast)

# ---------------------------------------------------------------- vocabularies
ACTION_VERBS = {
    "achieved", "administered", "analyzed", "analysed", "built", "collaborated", "completed",
    "conducted", "coordinated", "created", "delivered", "designed", "developed", "directed",
    "drove", "established", "evaluated", "executed", "generated", "implemented", "improved",
    "increased", "initiated", "launched", "led", "maintained", "managed", "negotiated",
    "optimized", "optimised", "organized", "organised", "oversaw", "planned", "produced",
    "reduced", "resolved", "saved", "streamlined", "supervised", "trained", "transformed",
    "automated", "boosted", "accelerated", "spearheaded", "mentored", "migrated", "deployed",
}

SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "academic"],
    "skills": ["skills", "competencies", "proficiencies"],
    "projects": ["projects"],
    "certifications": ["certification", "certifications", "licenses"],
    "awards": ["awards", "accomplishments", "achievements", "honors"],
}

# Baseline skill vocabulary (already in normalized form: lowercase, no punctuation except + #).
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "mysql", "postgresql",
    "mongodb", "excel", "power bi", "tableau", "machine learning", "deep learning",
    "tensorflow", "pytorch", "scikit learn", "pandas", "numpy", "nlp", "docker", "kubernetes",
    "aws", "azure", "git", "linux", "react", "angular", "node js", "html", "css", "django",
    "flask", "spring", "agile", "scrum", "communication", "leadership", "teamwork",
    "problem solving", "project management", "customer service", "sales", "marketing",
    "accounting", "budgeting", "financial analysis", "auditing", "quickbooks", "sap",
    "data analysis", "statistics", "research", "teaching", "curriculum", "recruitment",
    "payroll", "social media", "seo", "photoshop", "autocad", "networking", "cybersecurity",
    "salesforce", "jira", "time management", "negotiation", "microsoft office", "forecasting",
]
