# CVision

**An explainable, AI-powered CV evaluator for job seekers.**
Upload a CV as a PDF and get an overall quality score with a transparent breakdown, specific fixes
(e.g. bullet points that lack measurable results), a role-fit score based on sentence embeddings, and a
skill-gap chart showing in-demand skills the CV is missing.

> Course project: AI Project, Track A (Application Development)
> Deliverable 1, Problem Definition and Data Foundations, covers Phase 1 and Phase 2
> Deliverable 2: Model Architecture and Empirical Evaluation (Phase 3 and Phase 4)

**Authors:** Aqsa Ehtesham (24L-3017) and M. Shehryar Waheed (24L-3023)

## Why this project exists

Most applications are filtered by applicant tracking systems (ATS) before a human reads them, and rejected
candidates almost never learn why. CVision turns a single PDF into a fast, explainable evaluation so
candidates can improve their CV *before* it costs them an interview. The full problem statement, target
audience and AI justification are in the technical report (`report/`).

**Design principle:** rules for what is measurable (length, sections, numbers in bullets) keep the score
transparent; machine learning (sentence embeddings, a learned bullet-quality model) handles meaning and
writing quality. Everything runs locally, with no external AI API at runtime.

## Repository structure

```
CVision/
├── README.md
├── .gitignore
├── requirements.txt
├── data_scripts/           # all Phase 2 code
│   ├── config.py           # paths, thresholds, vocabularies
│   ├── utils.py            # text cleaning helpers
│   ├── collect.py          # step 1: data collection + evidence log
│   ├── clean.py            # step 2: missing values, duplicates, outliers
│   ├── features.py         # step 3: normalisation, scaling, encoding, TF-IDF
│   ├── eda.py              # step 4: exploratory charts
│   └── run_all.py          # runs steps 1-4
├── tests/make_demo_data.py # tiny SYNTHETIC data, for testing the pipeline only
├── data/                   # raw + processed data (git-ignored, see below)
├── figures/                # EDA charts used in the report
├── logs/                   # machine-generated evidence (collection log, cleaning report, ...)
├── src/                    # Phase 3 and 4 code (dataset, models, training, tuning, evaluation, bias check)
├── models/                 # trained weights (download link below, not in Git)
└── report/                 # LaTeX source and compiled PDF
```

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| Resume Dataset (`Resume.csv`): 2,484 CVs from livecareer.com in 24 job categories | [Kaggle: snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | CV-quality features, writing and section analysis |
| LinkedIn Job Postings 2023-2024 (`postings.csv`): 123,849 job ads | [Kaggle: arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Role-fit scoring, skill-gap chart |

The raw files are **not committed** (see `.gitignore`). `collect.py` records the source URL, file size,
SHA-256 hash and row/column counts of whatever you download in `logs/collection_log.json`, which is the
evidence of collection. A random sample of 40,000 job postings (seed 42) is used to keep processing fast.

## How to run

**1. Set up**

```bash
git clone https://github.com/<username>/CVision.git
cd CVision
python -m venv .venv
pip install -r requirements.txt
```

Activate the virtual environment first if you use one: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux.

**2. Get the data (either option)**

- *Manual:* download both datasets from the Kaggle links above and put `Resume.csv` and `postings.csv`
  anywhere inside `data/raw/`.
- *Automatic:* put your `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`) and run
  `python data_scripts/collect.py --download`.

**3. Run the pipeline**

```bash
python data_scripts/run_all.py
```

Or run the steps individually: `collect.py`, then `clean.py`, then `features.py`, then `eda.py`.

**Testing without the real data** (synthetic data, never use it for results):

```powershell
# Windows PowerShell
$env:CVISION_DATA_DIR = "$PWD\demo_data"
python tests/make_demo_data.py
python data_scripts/run_all.py
```

```bash
# macOS / Linux
CVISION_DATA_DIR=/tmp/demo python tests/make_demo_data.py
CVISION_DATA_DIR=/tmp/demo python data_scripts/run_all.py
```

## Run Phase 3 and 4
```bash
python src/tune.py        # hyperparameter search
python src/train.py       # train the dual-branch net and ablation
python src/evaluate.py    # compare all 5 models
python src/bias_check.py  # subgroup bias audit
```

## Model Weights

The trained weights are hosted externally and are not stored in Git:

- **Dual Branch Net (`dual_branch_net.pt`, 5.3 MB):** [Download from Google Drive](https://drive.google.com/file/d/1jTenoWsozCcBcOnhtynoriBH7map32XI/view?usp=drive_link)

After downloading, place the file at `models/dual_branch_net.pt`.

## What each Phase 2 step does

**Collection (`collect.py`)**: locates or downloads both datasets, profiles them and writes the evidence log.

**Cleaning (`clean.py`)**
- *Missing values:* rows with no text/label (CVs) or no title/description (jobs) are dropped; categorical
  gaps become `Unknown`; numeric gaps are median-imputed and flagged; empty `skills_desc` gets a flag column.
- *Duplicates:* exact duplicates, then near-duplicates (identical after lowercasing and stripping punctuation);
  for jobs also repeated `job_id`s and re-posted ads.
- *Outliers:* CVs/jobs under a minimum word count or above an IQR fence (Q3 + 3xIQR) are removed; salaries are
  converted to yearly, impossible values removed, and the extremes winsorised (1st-99th percentile).
- Every dropped row is listed with its reason in `data/processed/*_audit.csv`.

**Feature strategy (`features.py`)**
- *Normalisation:* HTML/symbol stripping, Unicode normalisation, lowercasing; TF-IDF (1-2 grams) for CV text.
- *Engineered CV features:* length, sentence statistics, lexical diversity, numbers that are not years (a proxy
  for quantified impact), action-verb density, section coverage, skill coverage.
- *Scaling:* strongly skewed counts get `log1p`, then z-score scaling. Scalers and TF-IDF are fitted on the
  **training split only** (stratified 70/15/15 train/val/test) to avoid data leakage.
- *Encoding:* CV category to integer label; job work type and experience level to one-hot.

**EDA (`eda.py`)**: nine figures in `figures/`: class distribution, CV length before/after outlier removal,
feature correlations, missing values, job distributions, a skill demand vs. supply preview, and the effect of scaling.

## Results of the Phase 2 run

| | CVs | Job postings |
|---|---|---|
| Raw rows | 2,484 | 123,849 |
| After sampling | n/a | 40,000 |
| **After cleaning** | **2,439** | **37,162** |
| Train / val / test split | 1,707 / 366 / 366 (stratified by category) | n/a |

Main findings: the CV classes are imbalanced (largest-to-smallest ratio 5.67; BPO, Automobile and Agriculture
are under-represented); the job column `skills_desc` is empty for about 98% of postings, so skills must be
mined from the free-text description; and the median CV (751 words) is longer than the median posting
(482 words). Details and all figures are in the technical report.

## Technical report

The LaTeX source is `report/report.tex`; the compiled PDF is `report/report.pdf`. To rebuild it, run
`pdflatex report.tex` **twice** inside `report/` (the second run fills the table of contents), or upload
`report.tex` together with the `figures/` folder to Overleaf.

## Team contributions

| Member | Modules | Description |
|---|---|---|
| Aqsa Ehtesham (24L-3017) | `src/dataset.py`, `src/baseline_models.py`, `src/evaluate.py`, Phase 1 and 2 foundations | Data loading and stratified 70/15/15 split, baseline models (Dummy, Logistic Regression, Random Forest), evaluation metrics, ROC curves and confusion matrix |
| M. Shehryar Waheed (24L-3023) | `src/deep_model.py`, `src/train.py`, `src/tune.py`, `src/bias_check.py`, `src/quality_score.py` | PyTorch dual-branch network, training loop with AdamW and weighted loss, 10-trial hyperparameter tuning, Phase 4 bias audit |

Individual contributions are verifiable in the Git commit history (`git log --author`).

## Limitations (to address in later phases)

- Resume text in the CV dataset comes from livecareer.com templates, so it is cleaner than typical real-world CVs,
  and its 24 mostly non-technical occupations under-represent modern technical profiles.
- The skill list in `config.py` is a baseline vocabulary of 70 skills matched by exact keywords; later phases will
  extend it from job postings and use embeddings.
- Salary data is very sparse and a few implausible yearly values remain, so it is only a weak, optional feature.
- Only a 40,000-posting sample of the job data is used.
- No PDF parsing yet; that arrives with the web app.
