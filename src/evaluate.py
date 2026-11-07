from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
PREPROCESSOR_FILE = PROCESSED_DIR / "artifacts" / "preprocessors.joblib"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Primary model designated in advance (avoids test-set snooping)
PRIMARY_MODEL_SLUG = "dual_branch_net"


def get_class_names():
    """Loads target class names from saved preprocessors without re-fitting TF-IDF."""
    if PREPROCESSOR_FILE.exists():
        prep = joblib.load(PREPROCESSOR_FILE)
        return list(prep["label_encoder"].classes_)

    # Fallback to dataset split only if preprocessors not dumped yet
    from dataset import prepare_data_splits
    return prepare_data_splits()["classes"]


def load_all_predictions():
    """Loads all prediction npz files matching logs/preds_*.npz."""
    pred_files = sorted(list(LOGS_DIR.glob("preds_*.npz")))
    if not pred_files:
        raise FileNotFoundError(
            "No prediction files found in logs/. Run src/baseline_models.py "
            "and src/train.py first."
        )

    models_data = {}
    for f in pred_files:
        # Standardize slug: preds_dual_branch_net.npz -> dual_branch_net
        slug = f.stem.replace("preds_", "")
        data = np.load(f)

        y_prob = data["test_proba"]
        y_true = data["test_labels"]
        y_pred = data["test_preds"] if "test_preds" in data else np.argmax(y_prob, axis=1)

        models_data[slug] = {
            "display_name": slug.replace("_", " ").title(),
            "proba": y_prob,
            "preds": y_pred,
            "labels": y_true
        }
    return models_data


def save_per_class_and_confusion_pairs(slug, name, d, class_names):
    """Exports per-class breakdown CSV and top confused pairs for Phase 4 error analysis."""
    y_true = d["labels"]
    y_pred = d["preds"]

    # 1. Per-class Precision, Recall, F1 report
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    rep_path = LOGS_DIR / f"per_class_{slug}.csv"
    rep_df.to_csv(rep_path)
    print(f"[{name}] Per-class report saved to {rep_path.name}")

    # 2. Extract top 10 off-diagonal confused pairs
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # Exclude correct diagonal classifications

    top_indices = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:10], cm.shape))[0]

    confused_pairs = []
    for actual_idx, pred_idx in top_indices:
        count = int(cm[actual_idx, pred_idx])
        if count > 0:
            confused_pairs.append({
                "actual_class": class_names[actual_idx],
                "predicted_class": class_names[pred_idx],
                "misclassified_count": count
            })

    conf_path = LOGS_DIR / f"most_confused_pairs_{slug}.json"
    pd.DataFrame(confused_pairs).to_json(conf_path, indent=2, orient="records")
    print(f"[{name}] Top confused pairs saved to {conf_path.name}")


def compute_comprehensive_metrics(models_data, class_names):
    """Computes rigorous test metrics with zero silent error suppression."""
    n_classes = len(class_names)
    records = []

    for slug, d in models_data.items():
        name = d["display_name"]
        y_true = d["labels"]
        y_pred = d["preds"]
        y_prob = d["proba"]

        # Strict assertion: every model must output exactly n_classes probabilities
        assert y_prob.shape[1] == n_classes, (
            f"Model '{name}' output shape {y_prob.shape[1]} does not match expected {n_classes} classes."
        )

        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Strict calculation: let failure surface if math is invalid
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

        records.append({
            "slug": slug,
            "Model Architecture": name,
            "Accuracy": acc,
            "Macro Precision": prec_macro,
            "Macro Recall": rec_macro,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "OvR AUC-ROC": auc_ovr
        })

        # Generate per-class reports and error diagnostics
        save_per_class_and_confusion_pairs(slug, name, d, class_names)

    df = pd.DataFrame(records).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return df


def plot_comparative_roc_curves(models_data, metrics_df, class_names):
    """Plots comparative ROC curves. Legend AUC values strictly match the table."""
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))

    # Create mapping from slug to exact table AUC
    auc_lookup = dict(zip(metrics_df["slug"], metrics_df["OvR AUC-ROC"]))

    for slug, d in models_data.items():
        name = d["display_name"]
        y_true = d["labels"]
        y_prob = d["proba"]
        table_auc = auc_lookup[slug]

        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        valid_classes = 0

        for c in range(n_classes):
            y_true_c = (y_true == c).astype(int)
            if np.sum(y_true_c) > 0:
                fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
                valid_classes += 1

        if valid_classes > 0:
            mean_tpr /= valid_classes
            plt.plot(all_fpr, mean_tpr, lw=2, label=f"{name} (Macro AUC = {table_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("One-vs-Rest ROC Curves Comparison (15% Test Set)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = FIG_DIR / "fig11_roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"ROC Curves saved to {out_path.name}")


def plot_confusion_matrix(model_slug, d, class_names):
    """Plots clean, uncluttered 24-class Normalized Confusion Matrix."""
    y_true = d["labels"]
    y_pred = d["preds"]
    display_name = d["display_name"]

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))

    # include_values=False ensures the 24x24 matrix remains clean and legible
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=90,
        include_values=False
    )

    plt.title(f"Normalized Confusion Matrix ({display_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "fig12_confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {out_path.name}")


def export_latex_table(df):
    """Exports LaTeX table code formatted with booktabs."""
    table_df = df.drop(columns=["slug"])
    latex_code = table_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Empirical Model Comparison on the 15\\% Test Split",
        label="tab:model_comparison",
        column_format="lcccccc"
    )
    table_path = LOGS_DIR / "model_comparison_table.tex"
    table_path.write_text(latex_code)
    print(f"LaTeX table exported to {table_path.name}")


def main():
    class_names = get_class_names()
    print(f"Evaluating across {len(class_names)} classes.")

    models_data = load_all_predictions()
    print(f"Loaded predictions for models: {[d['display_name'] for d in models_data.values()]}")

    # 1. Compute empirical metrics and save per-class CSVs
    df = compute_comprehensive_metrics(models_data, class_names)
    print("\n--- Empirical Results Table ---")
    print(df.drop(columns=["slug"]).to_string(index=False))

    # 2. Plot ROC curves (synchronized legend)
    plot_comparative_roc_curves(models_data, df, class_names)

    # 3. Plot Confusion Matrix for designated primary model (or top baseline if deep model not run yet)
    target_slug = PRIMARY_MODEL_SLUG if PRIMARY_MODEL_SLUG in models_data else df.iloc[0]["slug"]
    plot_confusion_matrix(target_slug, models_data[target_slug], class_names)

    # 4. Export LaTeX Table
    export_latex_table(df)


if __name__ == "__main__":
    main()