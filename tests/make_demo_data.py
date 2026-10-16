"""Generate SMALL SYNTHETIC stand-ins for the two Kaggle files, with the same column names.

FOR PIPELINE TESTING ONLY - never use this data in the report or for results.
It deliberately injects missing values, duplicates and outliers so clean.py has work to do.

    CVISION_DATA_DIR=/tmp/cvision_demo python tests/make_demo_data.py
    CVISION_DATA_DIR=/tmp/cvision_demo python data_scripts/run_all.py
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_scripts"))
from config import RAW_DIR  # noqa: E402

rng = random.Random(7)
np.random.seed(7)

CATS = {
    "INFORMATION-TECHNOLOGY": ["python", "sql", "docker", "machine learning", "aws", "git", "java"],
    "ACCOUNTANT": ["excel", "auditing", "budgeting", "quickbooks", "financial analysis", "payroll"],
    "TEACHER": ["curriculum", "teaching", "communication", "leadership", "research"],
    "SALES": ["sales", "negotiation", "customer service", "salesforce", "marketing"],
    "HR": ["recruitment", "payroll", "communication", "leadership", "time management"],
}
VERBS = ["Managed", "Developed", "Led", "Improved", "Designed", "Reduced", "Coordinated", "Responsible for"]


def make_resume(cat):
    skills = rng.sample(CATS[cat], k=min(3, len(CATS[cat])))
    parts = [f"{cat.replace('-', ' ')} Professional Summary Dedicated professional seeking growth.",
             "Experience"]
    for _ in range(rng.randint(3, 9)):
        verb = rng.choice(VERBS)
        tail = rng.choice([f"by {rng.randint(5, 60)}% in {rng.randint(2012, 2022)}.",
                           "across several teams.", f"for {rng.randint(3, 40)} clients."])
        parts.append(f"{verb} {rng.choice(skills)} projects {tail}")
    parts += ["Education Bachelor degree 2015 * University", "Skills " + ", ".join(skills)]
    return " ".join(parts) * rng.randint(3, 6)


rows = []
for i in range(420):
    cat = rng.choice(list(CATS))
    rows.append({"ID": 10_000 + i, "Resume_str": make_resume(cat), "Resume_html": "<div>x</div>", "Category": cat})
df = pd.DataFrame(rows)
df.loc[5, "Resume_str"] = np.nan                      # missing text
df.loc[9, "Category"] = np.nan                        # missing label
df.loc[12, "Resume_str"] = "Too short."               # short outlier
df.loc[20, "Resume_str"] = "Experience " * 9000       # long outlier
df = pd.concat([df, df.iloc[30:40], df.iloc[50:55].assign(Resume_str=lambda d: d["Resume_str"].str.upper(),
                                                          ID=lambda d: d["ID"] + 5000)], ignore_index=True)
(RAW_DIR / "Resume").mkdir(parents=True, exist_ok=True)
df.to_csv(RAW_DIR / "Resume" / "Resume.csv", index=False)

# ------------------------------------------------------------------ job postings
titles = ["Data Analyst", "Software Engineer", "Accountant", "Sales Manager", "HR Specialist", "Teacher"]
levels = ["Entry level", "Mid-Senior level", "Associate", "Director", np.nan]
types = ["Full-time", "Contract", "Part-time", "Internship"]
all_skills = sorted({s for v in CATS.values() for s in v})
jobs = []
for i in range(1500):
    sk = rng.sample(all_skills, k=4)
    desc = f"<p>We are hiring. <b>Requirements:</b> experience with {', '.join(sk)}.</p><br/>" + \
           "<li>Collaborate with team members and deliver results.</li> " * rng.randint(3, 12)
    jobs.append({
        "job_id": 900_000 + i, "company_name": f"Company {i % 80}", "title": rng.choice(titles),
        "description": desc, "skills_desc": ", ".join(sk) if rng.random() > 0.4 else np.nan,
        "location": "Lahore, PK", "formatted_work_type": rng.choice(types),
        "formatted_experience_level": rng.choice(levels), "remote_allowed": rng.choice([1.0, np.nan]),
        "max_salary": rng.choice([np.nan, np.nan, 25.0, 60000, 90000, 150000, 99999999]),
        "min_salary": rng.choice([np.nan, 20.0, 50000, 80000]),
        "pay_period": rng.choice(["HOURLY", "YEARLY", "MONTHLY", np.nan]),
        "views": rng.choice([np.nan, 10, 55, 300, 2000]), "applies": rng.choice([np.nan, 1, 8, 40]),
    })
jd = pd.DataFrame(jobs)
jd.loc[3, "description"] = np.nan
jd = pd.concat([jd, jd.iloc[100:130]], ignore_index=True)          # duplicates
jd.to_csv(RAW_DIR / "postings.csv", index=False)
print(f"Demo data written to {RAW_DIR}  (resumes: {len(df)} rows, postings: {len(jd)} rows)")
