# CVision

**An explainable, AI-powered CV evaluator for job seekers.**
Upload a CV as a PDF and get an overall quality score with a transparent breakdown, specific fixes
(e.g. bullet points that lack measurable results), a role-fit score based on sentence embeddings, and a
skill-gap chart showing in-demand skills the CV is missing.

> Course project: AI Project, Track A (Application Development)
> Deliverable 1: Problem Definition and Data Foundations (Phase 1 & Phase 2)
> Deliverable 2: Model Architecture and Empirical Evaluation (Phase 3 & Phase 4)
> Deliverable 3: Final Deployment and Technical Report (Phase 5 & Phase 6)

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
├── app.py                  # Deliverable 3: zero-dependency full-stack server (http.server)
├── static/                 # Deliverable 3: React 18 single-page application
│   └── index.html          # dashboard UI with Tailwind, drag-and-drop, presets
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
├── figures/                # EDA charts & UI dashboard screenshot used in the report
├── logs/                   # machine-generated evidence (collection log, cleaning report, LaTeX tables)
├── src/                    # Phase 3, 4 and 5 code
│   ├── dataset.py          # stratified split & PyTorch Dataset
│   ├── deep_model.py       # CVisionDualBranchNet (1,321,624 parameters)
│   ├── baseline_models.py  # Dummy, Logistic Regression, Random Forest
│   ├── train.py            # AdamW training loop with weighted loss
│   ├── tune.py             # 10-configuration hyperparameter sweep
│   ├── evaluate.py         # 5-model comparative benchmark & ROC curves
│   ├── bias_check.py       # subgroup bias audit
│   ├── quality_score.py    # 4-factor 0-100 heuristic scoring engine
│   └── inference.py        # real-time PDF extraction & LinkedIn skill gap engine
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
git clone https://github.com/shehryarwaheed/CVision.git
cd CVision
python -m venv .venv
pip install -r requirements.txt
```

Activate the virtual environment first if you use one: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux.

**2. Launch the Web Application (Deliverable 3 Quickstart)**

The full-stack web application runs with zero external web dependencies:

```bash
python app.py
```
This automatically opens your default browser at `http://localhost:8000`. You can:
- **Upload / Drag & Drop:** Drop any multi-page PDF resume or browse local files.
- **Paste Raw Text:** Toggle the expandable text area to paste and evaluate unformatted resume text.
- **Real-Time 0–100 Quality Score:** View circular meters for overall quality plus 4 orthogonal sub-scores (Quantified Impact 30%, Section Completeness 30%, Action Verb Density 20%, Lexical Diversity 20%).
- **Top-3 Occupational Matching:** Inspect predicted job categories with calibrated softmax confidence levels.
- **Market Skill-Gap Matrix:** Compare candidate skills against LinkedIn job market demand with matched and missing badges.
- **Parsed Text Preview (Tab 4):** Review verbatim extracted PDF text to verify ATS multi-column parsing integrity.
- **Instant Viva Presets:** Click any of the three pre-computed demo profiles (*Senior Full-Stack Engineer*, *Financial Analyst*, *HR Coordinator*) for immediate viva demonstration.
- **Export Diagnostic JSON:** Download a full offline evaluation report (`CVision_Diagnostic_<Role>.json`).

**3. Get the data (for retraining or reproducing Phase 2–4)**

- *Manual:* download both datasets from the Kaggle links above and put `Resume.csv` and `postings.csv`
  anywhere inside `data/raw/`.
- *Automatic:* put your `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<username>\.kaggle\`) and run
  `python data_scripts/collect.py --download`.

**4. Run the pipeline**

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

## AI Bullet Rewriter ("CV Coach")

CVision includes an **AI Bullet Rewriter** presented as a guided chat popup ("CV Coach"). It looks and feels like an interactive chatbot (speech bubbles, typing indicator, quick-reply chips), but runs on a deterministic, scripted state machine. The LLM is invoked strictly to rewrite CV bullet points.

### Key Capabilities
- **Section-Aware Bullet Extraction:** Automatically extracts Experience and Projects bullet points from PDF uploads or pasted text, handling bullet glyphs (`•`, `-`, `*`), numbered lists, and multi-line wrapped text from messy PDF extraction.
- **Rule-Based Weakness Detection (No LLM):** Evaluates bullets instantly using deterministic heuristics:
  - Flags passive starters: *"responsible for"*, *"worked on"*, *"helped"*, *"assisted"*, *"involved in"*, *"duties included"*, etc.
  - Flags missing metrics (absence of numbers, percentages, currency, scale).
  - Flags bad length (<8 words or >30 words).
  - Flags bullets not starting with a strong action verb from the curated 55-verb vocabulary.
  - Assigns an initial impact score (1–10) and actionable explanation (`why_weak`).
- **REST Endpoints (`app.py`):**
  - `POST /api/rewrite`: Rewrites a single bullet into exactly 3 distinct styles (`Impact-Driven`, `Technical & Scaled`, `Concise Executive`) matching the requested tone (`impact`, `technical`, `concise`, `leadership`).
  - `POST /api/rewrite-all`: Concurrently rewrites all weak bullets via `ThreadPoolExecutor` reusing the single-bullet logic (zero duplication).
  - `POST /api/extract-bullets`: Extracts and classifies bullets from arbitrary text.
- **Robustness & Fallbacks:**
  - Fence stripping, schema validation, and automatic single retry.
  - In-memory session cache per `(bullet, role, tone)`.
  - Deterministic offline template rewriter fallback: works without API keys or internet connection during viva demos.
  - Supports `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY` via standard library `urllib` (zero external npm/pip dependencies).
- **Interactive UI (React 18 SPA):**
  - **Entry Point 1:** Floating launcher button (bottom-right) with a live weak-bullet badge count.
  - **Entry Point 2:** Scorecard recommendation banner card (*"{N} bullets can be improved → Chat with CV Coach"*).
  - **Entry Point 3:** Per-bullet *"✨ Improve"* buttons in the Experience & Projects inspector.
  - **Live Score Sync:** When the user clicks *"✨ Use this"*, the CV text updates, the scoring engine re-evaluates the resume in real time, and an animated before ➔ after score update is displayed (`68 ➔ 72 (+4 pts)`).
  - **Metric Placeholders:** `[X%]` and `[N users]` placeholders are highlighted with an amber chip reminding the user to enter their actual numbers.

### Running Unit Tests
```bash
# Run backend tests (bullet extraction, weak detection, /rewrite, caching, batch)
python -m unittest discover tests

# Run frontend state machine tests (chat reducer lifecycle, tone selection, acceptance flow)
node --test tests/test_chat_reducer.js
```

### Clean Removal Note
The feature is strictly additive. To completely remove the CV Coach popup and launcher from the UI, delete this single line from `static/index.html`:
```jsx
<CVCoachWidget isDark={isDark} activeData={activeData} onAcceptRewrite={handleAcceptRewrite} externalTrigger={coachTrigger} />
```

## Team contributions

| Member | Deliverables & Modules | Description |
|---|---|---|
| Aqsa Ehtesham (24L-3017) | **D1 & D2:** `src/dataset.py`, `src/baseline_models.py`, `src/evaluate.py`, Phase 1 & 2 foundations<br>**D3:** Fallback inference validation & report verification | Data loading and stratified 70/15/15 split, baseline models (Dummy, Logistic Regression, Random Forest), evaluation metrics, ROC curves and confusion matrix; Deliverable 3 fallback validation and report verification |
| M. Shehryar Waheed (24L-3023) | **D1 & D2:** `src/deep_model.py`, `src/train.py`, `src/tune.py`, `src/bias_check.py`, `src/quality_score.py`<br>**D3:** `app.py`, `src/inference.py`, `static/index.html` | PyTorch dual-branch network, AdamW training loop, 10-trial hyperparameter sweep, Phase 4 bias audit, full 5-model evaluation; Deliverable 3 full-stack deployment (`app.py`, `src/inference.py` PDF parsing and LinkedIn skill gap engine, React 18 SPA with drag-and-drop, raw text editor, OCR preview, viva presets), Phase 5 & 6 technical report authoring |

Individual contributions are verifiable in the Git commit history (`git log --author`).

## Limitations and Future Work

- Resume text in the CV dataset comes from livecareer.com templates, so it is cleaner than typical real-world CVs,
  and its 24 mostly non-technical occupations under-represent modern technical profiles.
- Tail classes with very few samples (e.g. BPO with 4 test CVs, Automobile with 2 test CVs) suffer from severe data
  scarcity despite class-weighted cross-entropy loss.
- The skill list uses exact keyword matching against the LinkedIn vocabulary; future work will incorporate contextual
  transformer embeddings (e.g., Sentence-BERT) for deeper semantic matching.
- Real-world multi-column PDFs with complex graphical tables can introduce formatting noise during text extraction,
  which the application mitigates by exposing the verbatim extracted text preview in Tab 4.
