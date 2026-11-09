import json
from pathlib import Path
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import ROOT, get_dataloaders
from deep_model import CVisionDualBranchNet

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Sets all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer):
    """Executes a single training epoch with correct sample-weighted loss."""
    model.train()
    total_loss = 0.0
    n_seen = 0
    all_preds, all_labels = [], []

    for x_text, x_num, targets in loader:
        x_text = x_text.to(DEVICE)
        x_num = x_num.to(DEVICE)
        targets = targets.to(DEVICE)
        batch_n = targets.size(0)

        optimizer.zero_grad()
        logits = model(x_text, x_num)
        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_n
        n_seen += batch_n

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(n_seen, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    """Evaluates the model on validation or test sets."""
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_preds, all_labels, all_probs = [], [], []

    for x_text, x_num, targets in loader:
        x_text = x_text.to(DEVICE)
        x_num = x_num.to(DEVICE)
        targets = targets.to(DEVICE)
        batch_n = targets.size(0)

        logits = model(x_text, x_num)
        loss = criterion(logits, targets)

        total_loss += loss.item() * batch_n
        n_seen += batch_n
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(n_seen, 1)
    all_probs = np.vstack(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, all_probs, np.array(all_labels), np.array(all_preds)


def plot_learning_curves(history: dict, tag: str):
    """Generates Loss and Accuracy/F1 curves for the report."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss", lw=2, ms=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", lw=2, ms=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.set_title(f"Loss Curves ({tag.replace('_', ' ').title()})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Metric curve
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", lw=2, ms=4)
    ax2.plot(epochs, history["val_acc"], "r--s", label="Val Accuracy", lw=2, ms=4)
    ax2.plot(epochs, history["val_macro_f1"], "g-.^", label="Val Macro-F1", lw=2, ms=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title(f"Metrics Progress ({tag.replace('_', ' ').title()})", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_file = FIG_DIR / f"fig10_{tag}_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Learning curves saved to {out_file.name}")


def run_training(
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        dropout_rate: float = 0.4,
        hidden_text_dim: int = 256,
        hidden_num_dim: int = 32,
        hidden_fusion_dim: int = 128,
        use_num_branch: bool = True,
        num_epochs: int = 25,
        patience: int = 8,
        tag: str = "dual_branch_net",
        save_outputs: bool = True,
        evaluate_test: bool = True,
        random_state: int = 42,
) -> Tuple[float, Dict]:
    """
    Modular training loop callable by tune.py and ablation runs.
    Returns (best_val_macro_f1, history).
    """
    set_seed(random_state)

    # 1. Flexible unpack of get_dataloaders (handles both 4-item and 5-item signatures)
    ret = get_dataloaders(batch_size=batch_size, random_state=random_state)
    if len(ret) == 5:
        train_loader, val_loader, test_loader, class_weights, data = ret
    else:
        train_loader, val_loader, test_loader, data = ret
        class_weights = data.get("class_weights", None)

    n_classes = len(data["classes"])

    # 2. Instantiate Model
    model = CVisionDualBranchNet(
        tfidf_dim=5000,
        num_features=13,
        num_classes=n_classes,
        hidden_text_dim=hidden_text_dim,
        hidden_num_dim=hidden_num_dim,
        hidden_fusion_dim=hidden_fusion_dim,
        dropout_rate=dropout_rate,
        use_num_branch=use_num_branch,
    ).to(DEVICE)

    # 3. Loss & Optimizer
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    bad_epochs = 0
    best_model_state = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_macro_f1": []
    }

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_f1, _, _, _ = evaluate_epoch(model, val_loader, criterion)

        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_macro_f1"].append(va_f1)

        if save_outputs:
            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                f"Val Loss: {va_loss:.4f} Acc: {va_acc:.3f} Macro-F1: {va_f1:.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}"
            )
        # Checkpoint selection based on validation loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_f1 = va_f1
            best_epoch = epoch
            bad_epochs = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if save_outputs:
                    print(f"[{tag}] Early stopping triggered at epoch {epoch}")
                break

    # 4. Save Artifacts (only if save_outputs=True)
    if save_outputs and best_model_state is not None:
        checkpoint_path = MODELS_DIR / f"{tag}.pt"
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": best_model_state,
            "val_loss": best_val_loss,
            "val_macro_f1": best_val_f1,
            "use_num_branch": use_num_branch,
        }, checkpoint_path)
        print(f"[{tag}] Best checkpoint saved to {checkpoint_path.name} (Val Loss: {best_val_loss:.4f})")

        # Save plot and CSV history
        plot_learning_curves(history, tag)
        pd.DataFrame(history).to_csv(LOGS_DIR / f"train_history_{tag}.csv", index_label="epoch")
        with open(LOGS_DIR / f"training_history_{tag}.json", "w") as f:
            json.dump(history, f, indent=2)

    # 5. Evaluate on 15% Unseen Test Set (strictly once on final run)
    if evaluate_test and best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_loss, test_acc, test_f1, test_probs, test_labels, test_preds = evaluate_epoch(
            model, test_loader, criterion
        )
        test_auc = roc_auc_score(test_labels, test_probs, multi_class="ovr", average="macro")

        print(
            f"[{tag}] Test Performance: Acc={test_acc:.4f} | "
            f"Macro-F1={test_f1:.4f} | OvR-AUC={test_auc:.4f}"
        )

        # Save prediction npz for evaluate.py
        pred_path = LOGS_DIR / f"preds_{tag}.npz"
        np.savez(
            pred_path,
            test_proba=test_probs,
            test_preds=test_preds,
            test_labels=test_labels
        )
        print(f"[{tag}] Predictions exported to {pred_path.name}")

    return best_val_f1, history


def main():
    print("=" * 65)
    print("Training Primary Architecture: Dual-Branch Feature-Fusion Network")
    print("=" * 65)

    # 1. Train the primary multimodal model
    run_training(
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        dropout_rate=0.4,
        hidden_text_dim=256,
        hidden_num_dim=32,
        hidden_fusion_dim=128,
        use_num_branch=True,
        num_epochs=25,
        patience=8,
        tag="dual_branch_net",
        save_outputs=True,
        evaluate_test=True,
    )

    # 2. Optional: Train the text-only ablation model (for the report)
    print("\n" + "=" * 65)
    print("Training Ablation Architecture: Text-Only Branch (No Structural Features)")
    print("=" * 65)
    run_training(
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        dropout_rate=0.4,
        hidden_text_dim=256,
        use_num_branch=False,
        num_epochs=25,
        patience=8,
        tag="text_only_ablation",
        save_outputs=True,
        evaluate_test=True,
    )


if __name__ == "__main__":
    main()