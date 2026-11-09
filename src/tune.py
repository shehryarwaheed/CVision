import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from dataset import ROOT
from train import run_training

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_tuning_experiments():
    """Defines systematic trials varying one parameter axis at a time."""
    return [
        {"trial_id": 1, "desc": "Baseline", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 2, "desc": "High LR (1e-2)", "lr": 1e-2, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 3, "desc": "Low LR (1e-4)", "lr": 1e-4, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 4, "desc": "Small Batch (16)", "lr": 1e-3, "batch_size": 16, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 5, "desc": "Large Batch (64)", "lr": 1e-3, "batch_size": 64, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 6, "desc": "Low Dropout (0.2)", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.2, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 7, "desc": "High Dropout (0.6)", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.6, "weight_decay": 1e-4, "hidden_text_dim": 256},
        {"trial_id": 8, "desc": "No Weight Decay", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 0.0, "hidden_text_dim": 256},
        {"trial_id": 9, "desc": "Strong Weight Decay (1e-2)", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 1e-2, "hidden_text_dim": 256},
        {"trial_id": 10, "desc": "Compact Text (128)", "lr": 1e-3, "batch_size": 32, "dropout_rate": 0.4, "weight_decay": 1e-4, "hidden_text_dim": 128},
    ]


def export_tuning_latex_table(results_list, best_id):
    """Exports clean, booktabs-compliant LaTeX table with scientific formatting."""
    df_export = pd.DataFrame(results_list).copy()

    # Distinguish best_epoch from stopped_epoch
    df_export["Epochs (Best / Stop)"] = df_export.apply(
        lambda r: f"{r['best_epoch']} / {r['stopped_epoch']}", axis=1
    )

    df_export["Status"] = df_export["trial_id"].apply(
        lambda tid: "\\textbf{Selected}" if tid == best_id else "Tested"
    )

    # Scientific notation formatting
    df_export["Learning Rate"] = df_export["lr"].map(lambda v: f"{v:.0e}" if v > 0 else "0")
    df_export["Weight Decay"] = df_export["weight_decay"].map(lambda v: f"{v:.0e}" if v > 0 else "0")
    df_export["Batch Size"] = df_export["batch_size"]
    df_export["Dropout"] = df_export["dropout_rate"]
    df_export["Configuration"] = df_export["desc"]
    df_export["Val Macro-F1"] = df_export["val_macro_f1"]

    cols = ["Configuration", "Learning Rate", "Batch Size", "Dropout", "Weight Decay", "Epochs (Best / Stop)", "Val Macro-F1", "Status"]
    table_tex = df_export[cols].to_latex(
        index=False,
        float_format="%.4f",
        caption="Hyperparameter Exploration Grid and Validation Performance",
        label="tab:hyperparameter_tuning",
        column_format="lccccccr"
    )

    out_file = LOGS_DIR / "tuning_table.tex"
    out_file.write_text(table_tex)
    print(f"LaTeX tuning table exported to {out_file.name}")


def main():
    trials = get_tuning_experiments()
    print("=" * 70)
    print(f"Starting Phase 3 Hyperparameter Tuning: {len(trials)} trials scheduled")
    print("Settings: max_epochs=25, patience=8")
    print("=" * 70)

    results = []
    start_all = time.time()

    for t in trials:
        tid = t["trial_id"]
        desc = t["desc"]
        print(f"\n[Trial {tid:02d}/{len(trials):02d}] {desc}: LR={t['lr']:.0e}, BS={t['batch_size']}, Dropout={t['dropout_rate']}, WD={t['weight_decay']:.0e}")

        t0 = time.time()
        best_val_f1, history = run_training(
            lr=t["lr"],
            weight_decay=t["weight_decay"],
            batch_size=t["batch_size"],
            dropout_rate=t["dropout_rate"],
            hidden_text_dim=t["hidden_text_dim"],
            use_num_branch=True,
            num_epochs=25,
            patience=8,
            save_outputs=False,
            evaluate_test=False,
            random_state=42
        )
        elapsed = time.time() - t0

        best_epoch = int(np.argmin(history["val_loss"]) + 1)
        stopped_epoch = len(history["train_loss"])
        best_val_loss = float(min(history["val_loss"]))

        record = {
            **t,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "val_loss": best_val_loss,
            "val_macro_f1": float(best_val_f1),
            "time_seconds": round(elapsed, 1)
        }
        results.append(record)
        print(f"--> Val Loss: {best_val_loss:.4f} | Val Macro-F1: {best_val_f1:.4f} (Best at {best_epoch}, stopped at {stopped_epoch} in {elapsed:.1f}s)")

    total_time = time.time() - start_all
    df_sorted = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False).reset_index(drop=True)
    best_trial = df_sorted.iloc[0]
    best_id = int(best_trial["trial_id"])

    print("\n" + "=" * 70)
    print("Hyperparameter Tuning Completed!")
    print(f"Total Tuning Time: {total_time:.1f} seconds")
    print(f"Optimal Configuration: Trial {best_id} ({best_trial['desc']})")
    print(f"Best Val Macro-F1: {best_trial['val_macro_f1']:.4f} (Val Loss: {best_trial['val_loss']:.4f})")
    print("=" * 70)

    # 1. Export CSV Evidence Log
    csv_path = LOGS_DIR / "tuning_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"CSV results log saved to {csv_path.name}")

    # 2. Export JSON Log
    json_path = LOGS_DIR / "hyperparameter_tuning.json"
    with open(json_path, "w") as f:
        json.dump({
            "optimal_trial_id": best_id,
            "optimal_parameters": best_trial.to_dict(),
            "all_trials": results
        }, f, indent=2)
    print(f"JSON log saved to {json_path.name}")

    # 3. Export Formatted LaTeX Table in Trial Order
    export_tuning_latex_table(results, best_id)


if __name__ == "__main__":
    main()