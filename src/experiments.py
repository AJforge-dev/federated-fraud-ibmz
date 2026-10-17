"""Comprehensive Experiment Runner for Federated Fraud Detection on IBM Z.

Executes all 5 required benchmark experiments:
  1. Centralized baseline (pooled training data)
  2. Isolated baseline (per-bank models evaluated on shared test set)
  3. Federated Learning (FedAvg, 15 rounds, 2 local epochs/round)
  4. Federated Learning + Differential Privacy (clipping C=1.0, noise in [0, 0.1, 0.5, 1.0, 2.0])
  5. Imbalance Handling Comparison (class weighting alone vs local bank-level SMOTE)

Saves all metrics to results/metrics.csv, audit logs to results/audit_log.jsonl,
and visualization plots to results/*.png.
"""

import copy
import os
import random
import time
from typing import Any, Dict, List, Tuple

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import (
    BankPartition,
    get_dataloader,
    load_dataset,
    partition_data_non_iid,
    prepare_train_test_split,
)
from src.federated import run_federated_averaging
from src.model import (
    FraudMLP,
    compute_pos_weight,
    evaluate_model,
    get_loss_criterion,
    train_single_model,
)
from src.privacy import AuditLogger


def set_seed(seed: int = 42) -> None:
    """Enforces absolute reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_auprc_vs_rounds(
    round_histories: Dict[str, List[Dict[str, Any]]],
    output_path: str = "results/auprc_vs_rounds.png",
) -> None:
    """Plots AUPRC convergence over communication rounds for federated configurations."""
    plt.figure(figsize=(9, 5))
    colors = {
        "FedAvg (Class-Weighted)": "#1f77b4",
        "FedAvg + DP (σ=0.1)": "#2ca02c",
        "FedAvg + DP (σ=0.5)": "#ff7f0e",
        "FedAvg + SMOTE (Local)": "#9467bd",
    }

    for name, history in round_histories.items():
        rounds = [item["round"] for item in history]
        auprc_vals = [item["test_auprc"] for item in history]
        color = colors.get(name, None)
        plt.plot(
            rounds,
            auprc_vals,
            marker="o",
            linewidth=2,
            label=name,
            color=color,
        )

    plt.title(
        "Federated Learning Convergence (Shared Held-Out Test Set)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test AUPRC", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(range(1, 16))
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Convergence curve saved to {output_path}")


def plot_privacy_vs_utility(
    dp_results: List[Dict[str, Any]],
    output_path: str = "results/privacy_vs_utility.png",
) -> None:
    """Plots the trade-off between Differential Privacy noise multiplier and utility (Recall & AUPRC)."""
    noises = [res["noise_multiplier"] for res in dp_results]
    auprcs = [res["auprc"] for res in dp_results]
    recalls = [res["recall"] for res in dp_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_auprc = "#1f77b4"
    ax1.set_xlabel(
        "DP Noise Multiplier (σ)", fontsize=12, fontweight="medium"
    )
    ax1.set_ylabel("Test AUPRC", color=color_auprc, fontsize=12)
    line1 = ax1.plot(
        noises,
        auprcs,
        color=color_auprc,
        marker="s",
        linewidth=2.5,
        label="Test AUPRC",
    )
    ax1.tick_params(axis="y", labelcolor=color_auprc)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_recall = "#d62728"
    ax2.set_ylabel("Test Recall", color=color_recall, fontsize=12)
    line2 = ax2.plot(
        noises,
        recalls,
        color=color_recall,
        marker="^",
        linewidth=2.5,
        linestyle="--",
        label="Test Recall",
    )
    ax2.tick_params(axis="y", labelcolor=color_recall)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    plt.title(
        "Privacy vs. Utility Tradeoff (Differential Privacy Clipping C=1.0)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Privacy vs Utility plot saved to {output_path}")


def plot_confusion_matrices(
    methods_dict: Dict[str, Dict[str, Any]],
    output_path: str = "results/confusion_matrices.png",
) -> None:
    """Renders visual confusion matrices comparing key models."""
    n_models = len(methods_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, methods_dict.items()):
        cm = np.array(metrics["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=10, fontweight="bold")

        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,d}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Confusion matrices saved to {output_path}")


def run_all_experiments(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Executes the full suite of 5 benchmark experiments."""
    start_time = time.time()
    set_seed(42)
    device = torch.device("cpu")
    print(f"[EXPERIMENTS] Running on CPU (Device: {device})")

    audit_logger = AuditLogger("results/audit_log.jsonl")

    # 1. Data loading & preprocessing
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test, _ = prepare_train_test_split(
        df, test_size=0.2, random_state=42
    )
    bank_partitions = partition_data_non_iid(
        X_train, y_train, random_state=42
    )

    all_records: List[Dict[str, Any]] = []
    round_histories_for_plot: Dict[str, List[Dict[str, Any]]] = {}
    cm_dict_for_plot: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # EXPERIMENT 1: Centralized Baseline (Pooled Data)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: Centralized Baseline (Upper Bound)")
    print("=" * 78)
    set_seed(42)
    central_model = FraudMLP(input_dim=X_train.shape[1])
    central_loader = get_dataloader(
        X_train, y_train, batch_size=1024, shuffle=True
    )
    pos_weight_central = compute_pos_weight(y_train)
    central_criterion = get_loss_criterion(pos_weight_central, device=device)
    central_optimizer = torch.optim.Adam(central_model.parameters(), lr=0.001)

    print(f"Training Centralized Model on all {len(X_train):,} pooled rows for 10 epochs...")
    train_single_model(
        central_model,
        central_loader,
        central_criterion,
        central_optimizer,
        epochs=10,
        device=device,
    )
    eval_central = evaluate_model(central_model, X_test, y_test, device=device)
    eval_central["Method"] = "Centralized Baseline (Pooled)"
    all_records.append(eval_central)
    cm_dict_for_plot["Centralized"] = eval_central
    print(
        f"Centralized Results -> AUPRC: {eval_central['auprc']:.4f} | Recall: {eval_central['recall']:.4f} | F1: {eval_central['f1']:.4f}"
    )

    # =========================================================================
    # EXPERIMENT 2: Isolated Baseline (Per-Bank Private Models)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Isolated Baselines (Siloed Data)")
    print("=" * 78)
    isolated_metrics: List[Dict[str, Any]] = []

    for key, partition in bank_partitions.items():
        set_seed(42 + partition.bank_id)
        iso_model = FraudMLP(input_dim=X_train.shape[1])
        iso_loader = get_dataloader(
            partition.X, partition.y, batch_size=1024, shuffle=True
        )
        pos_weight_iso = compute_pos_weight(partition.y)
        iso_criterion = get_loss_criterion(pos_weight_iso, device=device)
        iso_optimizer = torch.optim.Adam(iso_model.parameters(), lr=0.001)

        print(
            f"Training Isolated Model for {partition.name} ({partition.size:,} rows, {partition.fraud_count} frauds)..."
        )
        train_single_model(
            iso_model,
            iso_loader,
            iso_criterion,
            iso_optimizer,
            epochs=10,
            device=device,
        )
        eval_iso = evaluate_model(iso_model, X_test, y_test, device=device)
        eval_iso["Method"] = f"Isolated - {partition.name}"
        isolated_metrics.append(eval_iso)
        all_records.append(eval_iso)
        print(
            f"  {partition.name} -> AUPRC: {eval_iso['auprc']:.4f} | Recall: {eval_iso['recall']:.4f} | F1: {eval_iso['f1']:.4f}"
        )

    # Calculate Weighted Average of Isolated Baselines
    weights = [p.size for p in bank_partitions.values()]
    total_w = sum(weights)
    weighted_mean_auprc = sum(
        m["auprc"] * w for m, w in zip(isolated_metrics, weights)
    ) / total_w
    weighted_mean_recall = sum(
        m["recall"] * w for m, w in zip(isolated_metrics, weights)
    ) / total_w
    weighted_mean_f1 = sum(
        m["f1"] * w for m, w in zip(isolated_metrics, weights)
    ) / total_w
    weighted_mean_prec = sum(
        m["precision"] * w for m, w in zip(isolated_metrics, weights)
    ) / total_w

    eval_iso_mean = {
        "Method": "Isolated Baseline (Weighted Mean)",
        "precision": weighted_mean_prec,
        "recall": weighted_mean_recall,
        "f1": weighted_mean_f1,
        "auprc": weighted_mean_auprc,
        "confusion_matrix": [[0, 0], [0, 0]],
        "tn": int(
            np.mean([m["tn"] for m in isolated_metrics])
        ),
        "fp": int(
            np.mean([m["fp"] for m in isolated_metrics])
        ),
        "fn": int(
            np.mean([m["fn"] for m in isolated_metrics])
        ),
        "tp": int(
            np.mean([m["tp"] for m in isolated_metrics])
        ),
        "test_size": len(y_test),
        "actual_fraud": int(np.sum(y_test)),
        "predicted_fraud": int(
            np.mean([m["predicted_fraud"] for m in isolated_metrics])
        ),
    }
    all_records.append(eval_iso_mean)
    cm_dict_for_plot["Isolated (Mean)"] = eval_iso_mean
    print(
        f"Isolated Weighted Mean -> AUPRC: {weighted_mean_auprc:.4f} | Recall: {weighted_mean_recall:.4f} | F1: {weighted_mean_f1:.4f}"
    )

    # =========================================================================
    # EXPERIMENT 3: Federated Learning (FedAvg, Manual PyTorch)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: Federated Learning (FedAvg - 15 Rounds, 2 Local Epochs)")
    print("=" * 78)
    set_seed(42)
    fed_model, fed_history = run_federated_averaging(
        bank_partitions=bank_partitions,
        X_test=X_test,
        y_test=y_test,
        rounds=15,
        local_epochs=2,
        batch_size=1024,
        lr=0.001,
        clip_norm=None,
        noise_multiplier=0.0,
        audit_logger=audit_logger,
        experiment_tag="FedAvg-Standard",
        device=device,
        verbose=True,
    )
    eval_fed = evaluate_model(fed_model, X_test, y_test, device=device)
    eval_fed["Method"] = "Federated (FedAvg, 15 Rounds)"
    all_records.append(eval_fed)
    round_histories_for_plot["FedAvg (Class-Weighted)"] = fed_history
    cm_dict_for_plot["FedAvg"] = eval_fed
    print(
        f"FedAvg Final -> AUPRC: {eval_fed['auprc']:.4f} | Recall: {eval_fed['recall']:.4f} | F1: {eval_fed['f1']:.4f}"
    )

    # =========================================================================
    # EXPERIMENT 4: Federated Learning + Differential Privacy (DP-FedAvg)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EXPERIMENT 4: Federated + Differential Privacy (Noise Multipliers: [0, 0.1, 0.5, 1.0, 2.0])")
    print("=" * 78)
    noise_multipliers = [0.0, 0.1, 0.5, 1.0, 2.0]
    dp_results: List[Dict[str, Any]] = []

    for sigma in noise_multipliers:
        set_seed(42)
        print(f"\nEvaluating DP-FedAvg with Noise Multiplier σ = {sigma} (Clip C=1.0)...")
        dp_model, dp_history = run_federated_averaging(
            bank_partitions=bank_partitions,
            X_test=X_test,
            y_test=y_test,
            rounds=15,
            local_epochs=2,
            batch_size=1024,
            lr=0.001,
            clip_norm=1.0,
            noise_multiplier=sigma,
            audit_logger=audit_logger,
            experiment_tag=f"DP-FedAvg-sigma_{sigma}",
            device=device,
            verbose=False,
        )
        eval_dp = evaluate_model(dp_model, X_test, y_test, device=device)
        eval_dp["noise_multiplier"] = sigma
        eval_dp["Method"] = f"FedAvg + DP (σ={sigma})"
        dp_results.append(eval_dp)
        all_records.append(eval_dp)
        print(
            f"  σ={sigma:<4} -> AUPRC: {eval_dp['auprc']:.4f} | Recall: {eval_dp['recall']:.4f} | F1: {eval_dp['f1']:.4f}"
        )

        if sigma == 0.1:
            round_histories_for_plot["FedAvg + DP (σ=0.1)"] = dp_history
            cm_dict_for_plot["FedAvg + DP (σ=0.1)"] = eval_dp
        elif sigma == 0.5:
            round_histories_for_plot["FedAvg + DP (σ=0.5)"] = dp_history

    plot_privacy_vs_utility(dp_results, "results/privacy_vs_utility.png")

    # =========================================================================
    # EXPERIMENT 5: Imbalance Handling (Class Weighting vs Local SMOTE)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EXPERIMENT 5: Imbalance Handling (Class Weighting vs Local Bank SMOTE)")
    print("=" * 78)
    print("Applying SMOTE strictly locally inside each bank's private partition...")
    smote_partitions: Dict[str, BankPartition] = {}

    for key, partition in bank_partitions.items():
        smote = SMOTE(random_state=42 + partition.bank_id)
        X_res, y_res = smote.fit_resample(partition.X, partition.y)
        smote_partitions[key] = BankPartition(
            bank_id=partition.bank_id,
            name=f"{partition.name} (SMOTE)",
            X=X_res,
            y=y_res,
            size=len(y_res),
            fraud_count=int(np.sum(y_res)),
            fraud_rate=float(np.mean(y_res) * 100.0),
        )
        print(
            f"  {partition.name}: Resampled from {partition.size:,} to {len(y_res):,} samples (50% fraud balance)."
        )

    set_seed(42)
    smote_model, smote_history = run_federated_averaging(
        bank_partitions=smote_partitions,
        X_test=X_test,
        y_test=y_test,
        rounds=15,
        local_epochs=2,
        batch_size=1024,
        lr=0.001,
        clip_norm=None,
        noise_multiplier=0.0,
        audit_logger=audit_logger,
        experiment_tag="FedAvg-SMOTE-Local",
        device=device,
        verbose=True,
    )
    eval_smote = evaluate_model(smote_model, X_test, y_test, device=device)
    eval_smote["Method"] = "FedAvg + SMOTE (Local)"
    all_records.append(eval_smote)
    round_histories_for_plot["FedAvg + SMOTE (Local)"] = smote_history
    cm_dict_for_plot["FedAvg + SMOTE"] = eval_smote
    print(
        f"FedAvg + SMOTE Results -> AUPRC: {eval_smote['auprc']:.4f} | Recall: {eval_smote['recall']:.4f} | F1: {eval_smote['f1']:.4f}"
    )

    # Save plots
    plot_auprc_vs_rounds(round_histories_for_plot, "results/auprc_vs_rounds.png")
    plot_confusion_matrices(cm_dict_for_plot, "results/confusion_matrices.png")

    # Compile and save final metrics DataFrame
    metrics_df = pd.DataFrame(all_records)
    # Reorder columns cleanly
    cols = [
        "Method",
        "precision",
        "recall",
        "f1",
        "auprc",
        "tn",
        "fp",
        "fn",
        "tp",
        "test_size",
        "actual_fraud",
        "predicted_fraud",
    ]
    # Filter to cols that exist
    cols = [c for c in cols if c in metrics_df.columns]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv("results/metrics.csv", index=False)
    print(f"\n[METRICS] All experiment metrics saved to results/metrics.csv")

    elapsed = time.time() - start_time
    print(f"\n[BENCHMARK COMPLETE] All 5 experiments completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes).")

    # Print summary table
    print_benchmark_summary(metrics_df)

    return metrics_df


def print_benchmark_summary(metrics_df: pd.DataFrame) -> None:
    """Prints a clean, presentation-ready comparison table."""
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON TABLE (HELD-OUT SHARED TEST SET)")
    print("=" * 90)
    header = f"{'Method / Experiment':<35} | {'AUPRC':>8} | {'Recall':>8} | {'Precision':>10} | {'F1 Score':>9}"
    print(header)
    print("-" * len(header))
    for _, row in metrics_df.iterrows():
        print(
            f"{row['Method']:<35} | {row['auprc']:>8.4f} | {row['recall']:>8.4f} | {row['precision']:>10.4f} | {row['f1']:>9.4f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_all_experiments()
