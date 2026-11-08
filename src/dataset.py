
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CVISION_DATA_DIR", ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
ART_DIR = PROCESSED_DIR / "artifacts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = PROCESSED_DIR / "split_indices.json"
PREPROCESSOR_FILE = ART_DIR / "preprocessors.joblib"

FEATURE_COLS = [
    "word_count", "sentence_count", "avg_sentence_len", "avg_word_len",
    "lexical_diversity", "metric_number_count", "percent_count",
    "numbers_per_100_words", "quantified_sentence_ratio",
    "action_verb_count", "action_verbs_per_100_words",
    "section_count", "skill_count"
]


class ResumeTorchDataset(Dataset):
    """PyTorch Dataset returning (tfidf_tensor, numerical_tensor, label)."""
    def __init__(self, x_tfidf, x_num, y):
        self.x_tfidf = torch.tensor(x_tfidf, dtype=torch.float32)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_tfidf[idx], self.x_num[idx], self.y[idx]


def compute_class_weights(y_train, n_classes):
    """Computes inverse-frequency class weights for PyTorch CrossEntropyLoss."""
    counts = np.bincount(y_train, minlength=n_classes)
    weights = len(y_train) / (n_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def load_resumes():
    """
    Loads resumes from resumes_clean.csv or raw/demo CSV.
    Ensures text, category, and structural features exist.
    """
    candidates = [
        PROCESSED_DIR / "resumes_clean.csv",
        RAW_DIR / "Resume" / "Resume.csv",
        RAW_DIR / "Resume.csv",
        ]
    df = None
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break

    if df is None:
        raise FileNotFoundError(
            f"No resume data found in {PROCESSED_DIR} or {RAW_DIR}. "
            "Run data_scripts/run_all.py or tests/make_demo_data.py first."
        )

    # Standardize column naming
    text_col = "text_clean" if "text_clean" in df.columns else ("Resume_str" if "Resume_str" in df.columns else "text_raw")
    cat_col = "category" if "category" in df.columns else "Category"

    df = df.dropna(subset=[text_col, cat_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[cat_col] = df[cat_col].astype(str).str.strip().str.title()

    # Provide baseline values for structural features if loading raw text
    for col in FEATURE_COLS:
        if col not in df.columns:
            df["word_count"] = df[text_col].map(lambda s: len(s.split()))
            df["sentence_count"] = df[text_col].map(lambda s: max(1, s.count(".") + s.count("!")))
            df["avg_sentence_len"] = df["word_count"] / df["sentence_count"]
            df["avg_word_len"] = 5.0
            df["lexical_diversity"] = 0.5
            df["metric_number_count"] = 5.0
            df["percent_count"] = 1.0
            df["numbers_per_100_words"] = 2.0
            df["quantified_sentence_ratio"] = 0.3
            df["action_verb_count"] = 8.0
            df["action_verbs_per_100_words"] = 3.0
            df["section_count"] = 4.0
            df["skill_count"] = 6.0
            break

    return df, text_col, cat_col


def prepare_data_splits(random_state=42, force_recompute=False):
    """
    Stratified 70/15/15 Train / Validation / Test split.
    Guarantees zero leakage: TF-IDF & StandardScaler are fitted ON TRAIN ONLY.
    Caches split indices in split_indices.json and preprocessors in artifacts/.
    """
    df, text_col, cat_col = load_resumes()

    # Label encode target
    le = LabelEncoder()
    y_all = le.fit_transform(df[cat_col])
    n_classes = len(le.classes_)

    # Safeguard: check minimum samples per class for a 3-way split
    counts = np.bincount(y_all)
    assert counts.min() >= 3, (
        f"Class too small to stratify 3 ways: class '{le.classes_[counts.argmin()]}' "
        f"has only {counts.min()} samples (minimum required is 3)."
    )

    # 1. Determine or Load Split Indices
    if SPLIT_FILE.exists() and not force_recompute:
        with open(SPLIT_FILE, "r") as f:
            indices = json.load(f)
        train_idx = [i for i in indices["train"] if i in df.index]
        val_idx = [i for i in indices["val"] if i in df.index]
        test_idx = [i for i in indices["test"] if i in df.index]
    else:
        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            df.index, test_size=0.30, stratify=y_all, random_state=random_state
        )
        # Second split: 15% val, 15% test
        y_temp = y_all[df.index.isin(temp_idx)]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=y_temp, random_state=random_state
        )

        # Save split indices for absolute reproducibility
        with open(SPLIT_FILE, "w") as f:
            json.dump({
                "train": [int(i) for i in train_idx],
                "val": [int(i) for i in val_idx],
                "test": [int(i) for i in test_idx],
            }, f, indent=2)

    # Partition dataframes
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]

    y_train = le.transform(train_df[cat_col])
    y_val = le.transform(val_df[cat_col])
    y_test = le.transform(test_df[cat_col])

    # 2. Fit TF-IDF on TRAIN ONLY (Exact Deliverable 1 settings)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True
    )
    x_train_tfidf = tfidf.fit_transform(train_df[text_col]).toarray()
    x_val_tfidf = tfidf.transform(val_df[text_col]).toarray()
    x_test_tfidf = tfidf.transform(test_df[text_col]).toarray()

    # 3. Fit Scaler on TRAIN ONLY
    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(train_df[FEATURE_COLS].fillna(0).astype(float))
    x_val_num = scaler.transform(val_df[FEATURE_COLS].fillna(0).astype(float))
    x_test_num = scaler.transform(test_df[FEATURE_COLS].fillna(0).astype(float))

    # 4. Save Fitted Preprocessors for Inference & Deliverable 3
    joblib.dump({
        "tfidf": tfidf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
    }, PREPROCESSOR_FILE)

    # 5. Compute Class Weights for Loss
    weights = compute_class_weights(y_train, n_classes)

    print(f"[Dataset] Stratified Split: Train={len(y_train)} (70%), Val={len(y_val)} (15%), Test={len(y_test)} (15%)")
    print(f"[Dataset] Classes: {n_classes}, TF-IDF vocab: {x_train_tfidf.shape[1]}, Numerical features: {x_train_num.shape[1]}")

    return {
        "train": (x_train_tfidf, x_train_num, y_train),
        "val": (x_val_tfidf, x_val_num, y_val),
        "test": (x_test_tfidf, x_test_num, y_test),
        "class_weights": weights,
        "label_encoder": le,
        "classes": list(le.classes_),
        "tfidf": tfidf,
        "scaler": scaler,
    }


def get_dataloaders(batch_size=32, random_state=42):
    data = prepare_data_splits(random_state=random_state)
    train_ds = ResumeTorchDataset(*data["train"])
    val_ds = ResumeTorchDataset(*data["val"])
    test_ds = ResumeTorchDataset(*data["test"])

    # In src/dataset.py -> get_dataloaders():
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, data["class_weights"], data


if __name__ == "__main__":
    train_loader, val_loader, test_loader, weights, data = get_dataloaders()
    print(f"Sample class weights (first 5): {weights[:5]}")
    print("dataset.py successfully validated and ready.")