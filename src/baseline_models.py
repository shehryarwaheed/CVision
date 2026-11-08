import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import ROOT, prepare_data_splits

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def concatenate_features(x_tfidf, x_num):
    """Concatenates 5000-d TF-IDF matrix with 13 scaled numerical metrics."""
    return np.hstack([x_tfidf, x_num])


def compute_safe_auc(y_true, probas, n_classes):
    """Safely computes One-vs-Rest macro AUC-ROC; returns 0.5 for dummy/constant cases."""
    try:
        # If probabilities don't cover all classes (e.g. DummyClassifier edge case)
        if probas.shape[1] < n_classes:
            padded = np.zeros((probas.shape[0], n_classes))
            padded[:, : probas.shape[1]] = probas
            probas = padded
        return float(roc_auc_score(y_true, probas, multi_class="ovr", average="macro"))
    except Exception:
        return 0.5000


def train_and_eval_baseline(model, name, x_train, y_train, x_val, y_val, x_test, y_test, n_classes):
    """Fits an estimator, calculates Phase 4 metrics (inc. OvR AUC-ROC), and saves probabilities."""
    print(f"\n--- Training {name} ---")
    model.fit(x_train, y_train)

    # Predicted classes
    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)

    # Predicted probabilities (required for AUC-ROC and ROC curves)
    val_proba = model.predict_proba(x_val)
    test_proba = model.predict_proba(x_test)

    val_auc = compute_safe_auc(y_val, val_proba, n_classes)
    test_auc = compute_safe_auc(y_test, test_proba, n_classes)

    metrics = {
        "model": name,
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_macro_f1": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
        "val_weighted_f1": float(f1_score(y_val, val_preds, average="weighted", zero_division=0)),
        "val_auc_ovr": val_auc,
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_macro_precision": float(precision_score(y_test, test_preds, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, test_preds, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, test_preds, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, test_preds, average="weighted", zero_division=0)),
        "test_auc_ovr": test_auc,
    }

    print(
        f"[{name}] Test Acc: {metrics['test_accuracy']:.4f} | "
        f"Test Macro-F1: {metrics['test_macro_f1']:.4f} | "
        f"Test OvR AUC: {metrics['test_auc_ovr']:.4f}"
    )

    # Save test probabilities for evaluate.py to plot comparative ROC curves
    clean_name = name.lower().replace(" ", "_")
    np.savez(
        LOGS_DIR / f"preds_{clean_name}.npz",
        test_proba=test_proba,
        test_preds=test_preds,
        test_labels=y_test,
        )

    # Save fitted model artifact to models/
    save_path = MODELS_DIR / f"baseline_{clean_name}.joblib"
    joblib.dump(model, save_path)
    print(f"Saved model to {save_path.name} and probabilities to logs/preds_{clean_name}.npz")

    return metrics


def main():
    data = prepare_data_splits()
    x_tr_tf, x_tr_num, y_train = data["train"]
    x_va_tf, x_va_num, y_val = data["val"]
    x_te_tf, x_te_num, y_test = data["test"]

    x_train = concatenate_features(x_tr_tf, x_tr_num)
    x_val = concatenate_features(x_va_tf, x_va_num)
    x_test = concatenate_features(x_te_tf, x_te_num)

    n_classes = len(data["classes"])
    results = {}

    # 1. Baseline 0: Dummy Baseline (Majority Class Floor)
    dummy = DummyClassifier(strategy="most_frequent")
    results["dummy"] = train_and_eval_baseline(
        dummy, "Dummy Baseline", x_train, y_train, x_val, y_val, x_test, y_test, n_classes
    )

    # 2. Baseline 1: Multinomial Logistic Regression
    logreg = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    results["logistic_regression"] = train_and_eval_baseline(
        logreg, "Logistic Regression", x_train, y_train, x_val, y_val, x_test, y_test, n_classes
    )

    # 3. Baseline 2: Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    results["random_forest"] = train_and_eval_baseline(
        rf, "Random Forest", x_train, y_train, x_val, y_val, x_test, y_test, n_classes
    )

    # Save summary report
    log_file = LOGS_DIR / "baseline_results.json"
    log_file.write_text(json.dumps(results, indent=2))
    print(f"\nAll baseline results and metrics saved to {log_file}")


if __name__ == "__main__":
    main()