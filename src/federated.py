"""Manual PyTorch Federated Learning (FedAvg) implementation without external frameworks.

Executes federated rounds across simulated banks with weighted averaging,
per-bank local training, differential privacy clipping and noise injection,
and continuous cryptographic audit logging.
"""

from collections import OrderedDict
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import BankPartition, get_dataloader
from src.model import (
    FraudMLP,
    compute_pos_weight,
    evaluate_model,
    get_loss_criterion,
)
from src.privacy import (
    AuditLogger,
    add_gaussian_noise,
    clip_update,
    compute_model_hash,
    compute_update_delta,
    compute_update_l2_norm,
)


def train_local_client(
    client_model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 2,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Trains a client model locally on its private partition for specified epochs."""
    client_model.train()
    client_model.to(device)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

    for _ in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = client_model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()


def aggregate_updates(
    client_deltas: List[OrderedDict[str, torch.Tensor]],
    weights: List[float],
) -> OrderedDict[str, torch.Tensor]:
    """Computes weighted average of client parameter updates.

    Formula:
      Delta_agg = sum_{k=1}^K (alpha_k * Delta_k)
    where alpha_k = n_k / sum(n_j).
    """
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    aggregated_delta = OrderedDict()
    first_delta = client_deltas[0]

    for key in first_delta.keys():
        aggregated_tensor = torch.zeros_like(first_delta[key])
        for delta, alpha in zip(client_deltas, normalized_weights):
            aggregated_tensor += alpha * delta[key]
        aggregated_delta[key] = aggregated_tensor

    return aggregated_delta


def apply_update_to_global_model(
    global_model: nn.Module,
    update_delta: OrderedDict[str, torch.Tensor],
) -> None:
    """Applies aggregated delta to global model in-place: w_global = w_global + delta."""
    current_state = global_model.state_dict()
    new_state = OrderedDict()
    for key, param in current_state.items():
        new_state[key] = param + update_delta[key].to(param.device)
    global_model.load_state_dict(new_state)


def run_federated_averaging(
    bank_partitions: Dict[str, BankPartition],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 15,
    local_epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 0.001,
    clip_norm: Optional[float] = None,
    noise_multiplier: float = 0.0,
    audit_logger: Optional[AuditLogger] = None,
    experiment_tag: str = "FedAvg",
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Executes full Federated Learning training across banks.

    Args:
      bank_partitions: Mapping of bank_id -> BankPartition.
      X_test: Shared held-out test features.
      y_test: Shared held-out test labels.
      rounds: Number of federated communication rounds (default: 15).
      local_epochs: Number of local epochs per round (default: 2).
      batch_size: Local batch size for training.
      lr: Local Adam optimizer learning rate.
      clip_norm: L2 norm threshold C for clipping updates (None disables clipping).
      noise_multiplier: Multiplier for central DP Gaussian noise (0.0 disables noise).
      audit_logger: Logger instance to record tamper-proof audit trails.
      experiment_tag: Tag for metadata tracking.
      device: Execution device (cpu).
      verbose: Whether to print per-round progress.

    Returns:
      (trained_global_model, round_history_metrics)
    """
    # Deterministic initialization of global model
    global_model = FraudMLP(input_dim=X_test.shape[1])
    global_model.to(device)

    # Pre-build local data loaders and class-weighted loss functions for each bank
    bank_loaders: Dict[str, DataLoader] = {}
    bank_criteria: Dict[str, nn.Module] = {}
    bank_sizes: List[int] = []
    bank_names: List[str] = []

    for key, partition in bank_partitions.items():
        bank_loaders[key] = get_dataloader(
            partition.X, partition.y, batch_size=batch_size, shuffle=True
        )
        pos_weight = compute_pos_weight(partition.y)
        bank_criteria[key] = get_loss_criterion(pos_weight, device=device)
        bank_sizes.append(partition.size)
        bank_names.append(key)

    round_history: List[Dict[str, Any]] = []

    if verbose:
        print(
            f"--- Starting {experiment_tag} ({rounds} rounds, {local_epochs} local epochs/round, DP Noise={noise_multiplier}) ---"
        )

    for r in range(1, rounds + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        client_deltas: List[OrderedDict[str, torch.Tensor]] = []
        update_norms: Dict[str, float] = {}

        # 1. Local Training at each simulated bank node
        for key in bank_names:
            partition = bank_partitions[key]
            # Client receives current global weights
            client_model = FraudMLP(input_dim=X_test.shape[1])
            client_model.load_state_dict(copy.deepcopy(global_state))

            # Train locally inside simulated bank perimeter
            train_local_client(
                client_model=client_model,
                train_loader=bank_loaders[key],
                criterion=bank_criteria[key],
                epochs=local_epochs,
                lr=lr,
                device=device,
            )

            # Compute update delta = w_local - w_global
            delta = compute_update_delta(
                client_model.state_dict(), global_state
            )

            # 2. Privacy clipping (if configured)
            if clip_norm is not None:
                clipped_delta, raw_norm = clip_update(delta, clip_norm)
                client_deltas.append(clipped_delta)
                update_norms[key] = raw_norm
            else:
                raw_norm = compute_update_l2_norm(delta)
                client_deltas.append(delta)
                update_norms[key] = raw_norm

        # 3. Weighted aggregation across banks
        aggregated_delta = aggregate_updates(client_deltas, weights=bank_sizes)

        # 4. Central differential privacy noise injection
        if noise_multiplier > 0.0 and clip_norm is not None:
            aggregated_delta = add_gaussian_noise(
                aggregated_delta=aggregated_delta,
                noise_multiplier=noise_multiplier,
                clip_norm=clip_norm,
                device=device,
            )

        # 5. Apply aggregated update to global model
        apply_update_to_global_model(global_model, aggregated_delta)

        # 6. Cryptographic fingerprint & evaluation on shared test set
        model_hash = compute_model_hash(global_model)
        eval_metrics = evaluate_model(
            global_model, X_test, y_test, device=device
        )

        round_record = {
            "round": r,
            "experiment": experiment_tag,
            "test_auprc": eval_metrics["auprc"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "test_f1": eval_metrics["f1"],
            "global_model_hash": model_hash,
            "update_norms": update_norms,
            "noise_multiplier": noise_multiplier,
        }
        round_history.append(round_record)

        # 7. Audit Logging
        if audit_logger is not None:
            audit_logger.log_round(
                round_idx=r,
                participating_banks=bank_names,
                update_norms=update_norms,
                noise_multiplier=noise_multiplier,
                global_model_hash=model_hash,
                test_auprc=eval_metrics["auprc"],
                additional_info={"experiment": experiment_tag},
            )

        if verbose:
            print(
                f"  Round {r:2d}/{rounds:2d} | Test AUPRC: {eval_metrics['auprc']:.4f} | "
                f"Recall: {eval_metrics['recall']:.4f} | F1: {eval_metrics['f1']:.4f} | "
                f"Model Hash: {model_hash[:12]}..."
            )

    return global_model, round_history
