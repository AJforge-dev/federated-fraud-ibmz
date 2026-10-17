"""Data pipeline for Federated Fraud Detection on IBM Z.

Handles Kaggle Credit Card Fraud dataset loading, synthetic generation fallback,
StandardScaler preprocessing, 80/20 stratified splitting, and non-IID bank partitioning.
"""

from dataclasses import dataclass
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class BankPartition:
    """Holds partitioned dataset for a single simulated bank."""

    bank_id: int
    name: str
    X: np.ndarray
    y: np.ndarray
    size: int
    fraud_count: int
    fraud_rate: float


class FraudDataset(Dataset):
    """PyTorch Dataset wrapper for tabular credit card fraud data."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def generate_synthetic_creditcard_data(
    output_path: str = "data/creditcard.csv",
    n_samples: int = 284807,
    n_fraud: int = 492,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates realistic synthetic creditcard.csv matching Kaggle's exact schema and statistical distributions.

    Used as an immediate runnable fallback if the user has not yet downloaded Kaggle's creditcard.csv.
    """
    print(
        f"[DATA NOTICE] Generating synthetic dataset ({n_samples:,} rows, {n_fraud} fraud cases, 30 features) at {output_path}..."
    )
    rng = np.random.default_rng(random_state)

    n_legit = n_samples - n_fraud

    # Time feature: seconds elapsed over 48 hours (0 to 172792)
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = rng.uniform(0, 172800, size=n_fraud)

    # PCA features V1..V28
    # Non-fraud: standard normal distributions with slight covariances
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud: shifted distributions on key fraud-predictive PCA components (e.g., V4, V11, V12, V14, V17)
    v_fraud = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    v_fraud[:, 3] += 3.5  # V4 often positively correlated with fraud
    v_fraud[:, 10] += 3.0  # V11 positively correlated
    v_fraud[:, 11] -= 3.5  # V12 negatively correlated
    v_fraud[:, 13] -= 4.0  # V14 heavily negatively correlated
    v_fraud[:, 16] -= 3.0  # V17 heavily negatively correlated

    # Amount feature: log-normal distributions (fraud often has slightly different spending profiles)
    amount_legit = np.exp(rng.normal(loc=3.2, scale=1.3, size=n_legit))
    amount_fraud = np.exp(rng.normal(loc=3.8, scale=1.4, size=n_fraud))

    # Assemble legit & fraud blocks
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    legit_df = pd.DataFrame(
        np.column_stack([time_legit, v_legit, amount_legit, np.zeros(n_legit)]),
        columns=cols,
    )
    fraud_df = pd.DataFrame(
        np.column_stack([time_fraud, v_fraud, amount_fraud, np.ones(n_fraud)]),
        columns=cols,
    )

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    # Shuffle all rows while maintaining seed
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA NOTICE] Synthetic dataset saved to {output_path}.")
    return df


def load_dataset(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Loads creditcard.csv from disk.

    If not found, displays instructions and invokes synthetic generator fallback.
    """
    if not os.path.exists(data_path):
        print("=" * 78)
        print(f"[DATA SETUP REQUIRED] File not found at '{data_path}'.")
        print("To use the official Kaggle dataset:")
        print("  1. Download 'creditcard.csv' from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  2. Place it in: {os.path.abspath(data_path)}")
        print("=" * 78)
        print(
            "Creating realistic synthetic fallback data so experiments remain 100% runnable right now..."
        )
        return generate_synthetic_creditcard_data(data_path)

    print(f"[DATA] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(
        f"[DATA] Loaded {len(df):,} transactions. Fraud count: {df['Class'].sum():,} ({df['Class'].mean() * 100:.3f}%)."
    )
    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Splits data into stratified train/test sets and fits StandardScaler on Time and Amount on train set only."""
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].values.astype(int)

    # Stratified 80/20 train/test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler strictly on training set to prevent data leakage
    scaler = StandardScaler()
    X_train_df[["Time", "Amount"]] = scaler.fit_transform(
        X_train_df[["Time", "Amount"]]
    )
    X_test_df[["Time", "Amount"]] = scaler.transform(
        X_test_df[["Time", "Amount"]]
    )

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler


def partition_data_non_iid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Dict[str, BankPartition]:
    """Partitions the training set across 4 simulated banks unevenly with distinct fraud rates (non-IID).

    Target Proportions:
      Bank 1 (Tier-1 Retail): 40% size, low fraud rate (~0.06%)
      Bank 2 (Commercial):     30% size, medium fraud rate (~0.14%)
      Bank 3 (Digital Fintech):20% size, high fraud rate (~0.26%)
      Bank 4 (High-Risk Acq):  10% size, critical fraud rate (~0.55%)
    """
    total_train = len(y_train)
    fraud_indices = np.where(y_train == 1)[0]
    legit_indices = np.where(y_train == 0)[0]

    total_fraud = len(fraud_indices)
    total_legit = len(legit_indices)

    rng = np.random.default_rng(random_state)
    shuffled_fraud = rng.permutation(fraud_indices)
    shuffled_legit = rng.permutation(legit_indices)

    # Size targets (40%, 30%, 20%, 10%)
    bank_proportions = [0.40, 0.30, 0.20, 0.10]
    bank_names = [
        "Bank 1 (Tier-1 Retail)",
        "Bank 2 (Commercial)",
        "Bank 3 (Digital Fintech)",
        "Bank 4 (High-Risk Merchant)",
    ]

    # Calculate exact sample counts summing to total_train
    target_sizes = [int(p * total_train) for p in bank_proportions]
    target_sizes[0] += total_train - sum(target_sizes)  # Adjust rounding

    # Skewed fraud distribution weights for non-IID characteristic
    # Higher concentration in Bank 3 and Bank 4
    fraud_weights = np.array([0.14, 0.24, 0.30, 0.32])
    fraud_counts = np.round(fraud_weights * total_fraud).astype(int)
    fraud_counts[0] += total_fraud - sum(fraud_counts)  # Adjust rounding

    partitions: Dict[str, BankPartition] = {}
    fraud_offset = 0
    legit_offset = 0

    for idx, (name, size, n_f) in enumerate(
        zip(bank_names, target_sizes, fraud_counts), start=1
    ):
        n_l = size - n_f

        b_fraud_idx = shuffled_fraud[fraud_offset : fraud_offset + n_f]
        b_legit_idx = shuffled_legit[legit_offset : legit_offset + n_l]

        fraud_offset += n_f
        legit_offset += n_l

        bank_indices = np.concatenate([b_fraud_idx, b_legit_idx])
        rng.shuffle(bank_indices)

        b_X = X_train[bank_indices]
        b_y = y_train[bank_indices]
        actual_fraud = int(np.sum(b_y))
        actual_rate = (actual_fraud / len(b_y)) * 100.0

        key = f"Bank_{idx}"
        partitions[key] = BankPartition(
            bank_id=idx,
            name=name,
            X=b_X,
            y=b_y,
            size=len(b_y),
            fraud_count=actual_fraud,
            fraud_rate=actual_rate,
        )

    print_bank_distribution(partitions, total_train, total_fraud)
    return partitions


def print_bank_distribution(
    partitions: Dict[str, BankPartition],
    total_train: int,
    total_fraud: int,
) -> None:
    """Prints a clear formatted summary table of non-IID bank partitions."""
    print("\n" + "=" * 78)
    print("SIMULATED NON-IID BANK PARTITIONS (TRAINING DATA)")
    print("=" * 78)
    header = f"{'Bank Identifier':<30} | {'Total Rows':>12} | {'Fraud Count':>11} | {'Fraud Rate (%)':>14}"
    print(header)
    print("-" * len(header))
    for p in partitions.values():
        print(
            f"{p.name:<30} | {p.size:>12,d} | {p.fraud_count:>11,d} | {p.fraud_rate:>13.3f}%"
        )
    print("-" * len(header))
    overall_rate = (total_fraud / total_train) * 100.0
    print(
        f"{'Total Pooled Training Set':<30} | {total_train:>12,d} | {total_fraud:>11,d} | {overall_rate:>13.3f}%"
    )
    print("=" * 78 + "\n")


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a PyTorch DataLoader for given features and labels."""
    dataset = FraudDataset(X, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )


if __name__ == "__main__":
    # Smoke test when running python -m src.data
    df_data = load_dataset()
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test_split(df_data)
    parts = partition_data_non_iid(X_tr, y_tr)
    print(f"Data pipeline ready! Test set shape: {X_te.shape}, Frauds: {y_te.sum()}")
