"""PyTorch MLP model definition, loss function with pos_weight, and evaluation metrics.

Implements input 30 -> hidden 64 -> hidden 32 -> output 1 architecture.
Evaluation computes Precision, Recall, F1, AUPRC, and Confusion Matrix.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FraudMLP(nn.Module):
    """Multi-Layer Perceptron for tabular credit card fraud detection.

    Architecture:
      Input (30) -> Linear(64) -> ReLU -> Dropout(0.2)
                 -> Linear(32) -> ReLU -> Dropout(0.2)
                 -> Linear(1)  -> Raw Logits
    """

    def __init__(self, input_dim: int = 30, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs raw logits for numerical stability with BCEWithLogitsLoss."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Applies sigmoid to logits to generate fraud probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def compute_pos_weight(y: np.ndarray) -> float:
    """Computes pos_weight ratio (negatives / positives) for BCEWithLogitsLoss."""
    num_neg = np.sum(y == 0)
    num_pos = max(1, np.sum(y == 1))
    return float(num_neg / num_pos)


def get_loss_criterion(
    pos_weight: Optional[float] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.BCEWithLogitsLoss:
    """Instantiates BCEWithLogitsLoss with optional pos_weight for class imbalance."""
    if pos_weight is not None:
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    return nn.BCEWithLogitsLoss()


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """Evaluates model on the shared test set.

    Computes: Precision, Recall, F1, AUPRC (Average Precision), and Confusion Matrix.
    Accuracy is deliberately omitted as a primary metric due to severe class imbalance.
    """
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    num_samples = len(X_test)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = torch.tensor(
                X_test[i : i + batch_size], dtype=torch.float32, device=device
            )
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)

    y_probs = np.concatenate(all_probs)
    y_pred = (y_probs >= threshold).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auprc = float(average_precision_score(y_test, y_probs))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_size": len(y_test),
        "actual_fraud": int(np.sum(y_test)),
        "predicted_fraud": int(np.sum(y_pred)),
    }


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Trains a model locally for a given number of epochs. Returns loss history."""
    model.train()
    model.to(device)
    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        loss_history.append(avg_loss)

    return loss_history
