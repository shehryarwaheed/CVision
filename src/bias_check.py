import argparse
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report
import torch

from dataset import ROOT, SPLIT_FILE, load_resumes, prepare_data_splits
from deep_model import CVisionDualBranchNet
from quality_score import compute_cv_quality_score

LOGS_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_predictions_and_metadata(tag: str = "dual_branch_net"):
    """Loads predictions and verifies strict index alignment with raw test documents."""
    pred_path = LOGS_DIR / f"preds_{tag}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Required predictions '{pred_path.name}' not found. "
            f"Train the model first using: python src/train.py"
        )

    data = np.load(pred_path)
    test_proba = data["test_proba"]
    test_labels = data["test_labels"]
    test_preds = data["test_preds"] if "test_preds" in data else np.argmax(test_proba, axis=1)

    splits_data = prepare_data_splits()
    le = splits_data["label_encoder"]
    class_names = splits_data["classes"]

    df, text_col, cat_col = load_resumes()
    with open(SPLIT_FILE, "r") as f:
        test_indices = json.load(f)["test"]

    test_df = df.loc[test_indices].copy().reset_index(drop=True)

    # CRITICAL ALIGNMENT ASSERTION: Verify row-by-row label alignment
    derived_labels = le.transform(test_df[cat_col].astype(str))
    assert (derived_labels == test_labels).all(), (
        "CRITICAL ERROR: Predictions in .npz are misaligned with test document indices!"
    )

    test_df["word_count"] = test_df[text_col].astype(str).map(lambda s: len(s.split()))
    confidence = np.max(test_proba, axis=1)

    return {
        "df": test_df,
        "y_true": test_labels,
        "y_pred": test_preds,
        "y_prob": test_proba,
        "confidence": confidence,
        "class_names": class_names,
        "word_counts": test_df["word_count"].to_numpy(),
        "text_col": text_col,
        "cat_col": cat_col,
        "splits_data": splits_data,
    }


def audit_class_representation_bias(meta):
    """Audits whether model performance degrades systematically on under-represented classes."""
    y_true = meta["y_true"]
    y_pred = meta["y_pred"]
    class_names = meta["class_names"]

    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    records = []
    for c_idx, c_name in enumerate(class_names):
        records.append({
            "class_idx": c_idx,
            "class_name": c_name,
            "test_support": int(rep[c_name]["support"]),
            "precision": float(rep[c_name]["precision"]),
            "recall": float(rep[c_name]["recall"]),
            "f1": float(rep[c_name]["f1-score"])
        })

    class_df = pd.DataFrame(records).sort_values(by="test_support", ascending=False).reset_index(drop=True)

    top_5 = class_df.head(5)
    bottom_5 = class_df.tail(5)

    maj_f1 = float(top_5["f1"].mean())
    min_f1 = float(bottom_5["f1"].mean())
    disparity = maj_f1 - min_f1

    corr, p_val = pearsonr(class_df["test_support"], class_df["f1"])

    summary = {
        "top_5_classes": top_5["class_name"].tolist(),
        "top_5_support_range": f"{top_5['test_support'].min()}--{top_5['test_support'].max()}",
        "majority_mean_f1": maj_f1,
        "bottom_5_classes": bottom_5["class_name"].tolist(),
        "bottom_5_support_range": f"{bottom_5['test_support'].min()}--{bottom_5['test_support'].max()}",
        "minority_mean_f1": min_f1,
        "f1_disparity": float(disparity),
        "support_f1_correlation": float(corr),
        "correlation_p_value": float(p_val)
    }

    print("\n--- 1. Class Representation Bias Audit ---")
    print(f"Majority (Top-5) Test Support: {summary['top_5_support_range']} | Mean F1: {maj_f1:.4f}")
    print(f"Minority (Bottom-5) Test Support: {summary['bottom_5_support_range']} | Mean F1: {min_f1:.4f}")
    print(f"Disparity (Majority Advantage): {disparity:.4f}")
    print(f"Support vs. F1 Pearson r: {corr:.3f} (p = {p_val:.4f})")

    return class_df, summary


def audit_document_length_bias(meta):
    """Audits performance across document length terciles using sample-equal bins."""
    words = meta["word_counts"]
    y_true = meta["y_true"]
    y_pred = meta["y_pred"]
    conf = meta["confidence"]

    p33, p66 = np.percentile(words, [33.3, 66.6])
    bins = [0, p33, p66, np.inf]
    labels = [f"Short (<{int(p33)}w)", f"Medium ({int(p33)}-{int(p66)}w)", f"Long (>{int(p66)}w)"]

    bin_indices = np.digitize(words, bins) - 1

    records = []
    for idx, label in enumerate(labels):
        mask = (bin_indices == idx)
        count = int(np.sum(mask))
        acc = float(accuracy_score(y_true[mask], y_pred[mask])) if count > 0 else 0.0
        mean_conf = float(np.mean(conf[mask])) if count > 0 else 0.0

        records.append({
            "Slice": label,
            "Sample Count": count,
            "Accuracy": acc,
            "Mean Confidence": mean_conf
        })

    length_df = pd.DataFrame(records)
    print("\n--- 2. Document Length Tercile Audit ---")
    print(length_df.to_string(index=False))

    return length_df


def audit_gender_demographic_proxy(meta, tag: str = "dual_branch_net"):
    """
    Evaluates sensitivity to gendered markers on the SAME model architecture.
    Infers architecture dimensions directly from checkpoint to prevent size mismatch crashes.
    """
    print("\n--- 3. Demographic Proxy: Gender & Role Perturbation Test ---")
    tfidf = meta["splits_data"]["tfidf"]
    test_df = meta["df"]
    text_col = meta["text_col"]

    # Collision-free map: "ms" excluded to prevent corrupting "MS in Computer Science"
    swap_map = {
        "mr": "mrs", "mrs": "mr", "sir": "madam", "madam": "sir",
        "waiter": "waitress", "waitress": "waiter",
        "actor": "actress", "actress": "actor",
        "chairman": "chairwoman", "chairwoman": "chairman",
        "he": "she", "she": "he",
        "his": "her", "her": "his",
        "him": "her"
    }

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in swap_map.keys()) + r")\b", re.IGNORECASE)
    swap_func = lambda t: pattern.sub(lambda m: swap_map[m.group(0).lower()], t)

    original_texts = test_df[text_col].astype(str).tolist()
    perturbed_texts = [swap_func(t) for t in original_texts]

    changed_count = sum(a != b for a, b in zip(original_texts, perturbed_texts))
    total_docs = len(original_texts)

    # Transform both texts through the training TF-IDF
    x_orig_sparse = tfidf.transform(original_texts)
    x_pert_sparse = tfidf.transform(perturbed_texts)

    # Count how many feature vectors actually changed (prunes out stopwords like he/she)
    diff = abs(x_orig_sparse - x_pert_sparse).sum(axis=1)
    feat_changed = int((np.asarray(diff).ravel() > 1e-9).sum())

    print(f"Raw Text Changed: {changed_count}/{total_docs} ({changed_count/total_docs:.1%})")
    print(f"TF-IDF Vectors Changed (Non-Stopwords): {feat_changed}/{total_docs} ({feat_changed/total_docs:.1%})")

    # Reconstruct the exact model from the checkpoint
    checkpoint_path = MODELS_DIR / f"{tag}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint '{checkpoint_path.name}' required for perturbation testing.")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    sd = checkpoint["model_state_dict"]
    has_num = "num_branch.0.weight" in sd

    model = CVisionDualBranchNet(
        tfidf_dim=5000,
        num_features=13,
        num_classes=len(meta["class_names"]),
        hidden_text_dim=sd["text_branch.0.weight"].shape[0],
        hidden_num_dim=sd["num_branch.0.weight"].shape[0] if has_num else 32,
        hidden_fusion_dim=sd["fusion_layer.0.weight"].shape[0],
        use_num_branch=has_num,
    ).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    x_num = meta["splits_data"]["test"][1]

    with torch.no_grad():
        x_orig_t = torch.tensor(x_orig_sparse.toarray(), dtype=torch.float32).to(DEVICE)
        x_pert_t = torch.tensor(x_pert_sparse.toarray(), dtype=torch.float32).to(DEVICE)
        x_num_t = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)

        orig_logits = model(x_orig_t, x_num_t)
        pert_logits = model(x_pert_t, x_num_t)

        probs_orig = torch.softmax(orig_logits, dim=-1).cpu().numpy()
        preds_orig = torch.argmax(orig_logits, dim=-1).cpu().numpy()
        preds_pert = torch.argmax(pert_logits, dim=-1).cpu().numpy()

    # Reproducibility sanity check with floating-point tolerance
    prob_match = np.allclose(probs_orig, meta["y_prob"], atol=1e-4)
    pred_match_rate = float(np.mean(preds_orig == meta["y_pred"]))
    assert prob_match or pred_match_rate >= 0.995, (
        f"CRITICAL ERROR: Reloaded model fails to reproduce saved predictions! Match rate: {pred_match_rate:.2%}"
    )

    flips = int(np.sum(preds_orig != preds_pert))
    flip_rate = float(flips / total_docs)

    print(f"Perturbation Flip Rate (Same PyTorch Model): {flip_rate:.2%} ({flips}/{total_docs} changed)")

    return {
        "text_changed": changed_count,
        "features_changed": feat_changed,
        "total_test_resumes": total_docs,
        "flips_count": flips,
        "flip_rate": flip_rate,
        "consistency_rate": 1.0 - flip_rate
    }


def audit_quality_score_bias(meta):
    """Evaluates the canonical rule-based quality score across length and categories."""
    df = meta["df"]
    words = meta["word_counts"]
    cat_col = meta["cat_col"]

    quality_scores = compute_cv_quality_score(df)
    r_val, p_val = pearsonr(words, quality_scores)

    # Group scores by category
    cat_means = df.assign(quality_score=quality_scores).groupby(cat_col)["quality_score"].mean().sort_values(ascending=False)

    print("\n--- 4. Canonical Quality Score Audit ---")
    print(f"Correlation (Word Count vs. Quality Score): r = {r_val:.3f} (p = {p_val:.4e})")
    print(f"Highest Mean Quality Category: {cat_means.index[0]} ({cat_means.iloc[0]:.1f}/100)")
    print(f"Lowest Mean Quality Category:  {cat_means.index[-1]} ({cat_means.iloc[-1]:.1f}/100)")

    return {
        "length_correlation_r": float(r_val),
        "length_correlation_p": float(p_val),
        "category_means": cat_means.to_dict()
    }


def plot_bias_figure(class_df, length_df, tag: str):
    """Generates dual-panel Figure 13 for the LaTeX report."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Support vs Per-Class F1
    x = class_df["test_support"]
    y = class_df["f1"]
    ax1.scatter(x, y, color="#1e40af", s=65, alpha=0.85, edgecolors="none")

    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", lw=1.8, label=f"Trend (slope = {z[0]:+.4f})")

    ax1.set_xlabel("Class Test Support (Sample Count)", fontsize=11)
    ax1.set_ylabel("Per-Class F1-Score", fontsize=11)
    ax1.set_title("A: Representation vs. Per-Class F1-Score", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    for _, row in class_df.iterrows():
        if row["f1"] < 0.60 or row["test_support"] == class_df["test_support"].max():
            ax1.annotate(row["class_name"], (row["test_support"], row["f1"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8, alpha=0.9)

    # Panel B: Length Slices Accuracy and Sample Count
    slices = length_df["Slice"]
    accs = length_df["Accuracy"]
    counts = length_df["Sample Count"]

    idx = np.arange(len(slices))
    bar_width = 0.45

    bars = ax2.bar(idx, accs, bar_width, color="#059669", alpha=0.85, label="Accuracy")
    ax2.set_xlabel("Resume Length Terciles", fontsize=11)
    ax2.set_ylabel("Classification Accuracy", fontsize=11)
    ax2.set_title("B: Accuracy Across Document Length Terciles", fontsize=12, fontweight="bold")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(slices, fontsize=10)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar, acc, cnt in zip(bars, accs, counts):
        h = bar.get_height()
        ax2.annotate(f"{acc:.3f}\n(n={cnt})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = FIG_DIR / f"fig13_{tag}_bias_check.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Bias figure exported to {out_file.name}")


def export_bias_latex_table(length_df, summary, gender_res, n_test: int, tag: str):
    """Exports clean, escaped LaTeX table avoiding unescaped math or raw % characters."""
    rows = [
        {
            "Subgroup Audit Dimension": "Majority Categories (Top 5)",
            "Test Support": summary["top_5_support_range"],
            "Accuracy / F1": f"{summary['majority_mean_f1']:.4f}",
            "Disparity Finding": "Baseline Reference"
        },
        {
            "Subgroup Audit Dimension": "Minority Categories (Bottom 5)",
            "Test Support": summary["bottom_5_support_range"],
            "Accuracy / F1": f"{summary['minority_mean_f1']:.4f}",
            "Disparity Finding": f"Delta-F1: {summary['f1_disparity']:.4f}"
        },
    ]

    for _, r in length_df.iterrows():
        rows.append({
            "Subgroup Audit Dimension": f"Length: {r['Slice']}",
            "Test Support": str(r["Sample Count"]),
            "Accuracy / F1": f"{r['Accuracy']:.4f}",
            "Disparity Finding": f"Confidence: {r['Mean Confidence']:.3f}"
        })

    rows.append({
        "Subgroup Audit Dimension": "Gender/Title Perturbation",
        "Test Support": f"{gender_res['features_changed']} feature-changed ({n_test} total)",
        "Accuracy / F1": f"{gender_res['consistency_rate']:.4f} consistency",
        "Disparity Finding": f"Flip Rate: {gender_res['flip_rate'] * 100:.2f} percent"
    })

    df_table = pd.DataFrame(rows)
    tex_code = df_table.to_latex(
        index=False,
        escape=True,
        caption="Subgroup Bias Audit Across Category Representation, Document Length, and Gender Proxy",
        label=f"tab:bias_audit_{tag}",
        column_format="llcl"
    )

    out_file = LOGS_DIR / f"bias_table_{tag}.tex"
    out_file.write_text(tex_code)
    print(f"LaTeX bias table exported to {out_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Bias Check & Error Diagnostics")
    parser.add_argument("--tag", type=str, default="dual_branch_net", help="Model checkpoint tag to audit")
    args = parser.parse_args()

    print("=" * 65)
    print(f"Executing Phase 4 Subgroup Bias Audit for: {args.tag}")
    print("=" * 65)

    meta = load_test_predictions_and_metadata(tag=args.tag)
    n_test = len(meta["y_true"])

    class_df, summary = audit_class_representation_bias(meta)
    length_df = audit_document_length_bias(meta)
    gender_res = audit_gender_demographic_proxy(meta, tag=args.tag)
    qs_res = audit_quality_score_bias(meta)

    plot_bias_figure(class_df, length_df, args.tag)
    export_bias_latex_table(length_df, summary, gender_res, n_test, args.tag)

    audit_report = {
        "model_tag": args.tag,
        "class_representation": summary,
        "length_terciles": length_df.to_dict(orient="records"),
        "gender_perturbation": gender_res,
        "quality_score_audit": qs_res
    }
    with open(LOGS_DIR / f"bias_report_{args.tag}.json", "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Full audit report saved to logs/bias_report_{args.tag}.json")
    print("=" * 65)


if __name__ == "__main__":
    main()