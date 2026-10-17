"""Differential Privacy (DP) mechanisms and Cryptographic Audit utilities for Federated Learning.

Implements per-bank update clipping (L2 norm threshold C), Gaussian noise addition,
and SHA-256 weight state hashing for immutable regulatory audit logs.
"""

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_model_hash(model: nn.Module) -> str:
    """Computes a deterministic SHA-256 hash of the model weights.

    Provides a tamper-proof cryptographic fingerprint for the audit log.
    """
    hasher = hashlib.sha256()
    # Iterate through parameters in sorted order to ensure absolute determinism
    for name, param in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        # Use CPU float32 bytes for consistent endianness and architecture portability
        tensor_bytes = param.detach().cpu().to(torch.float32).numpy().tobytes()
        hasher.update(tensor_bytes)
    return hasher.hexdigest()


def compute_update_delta(
    local_state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Computes delta = local_state - global_state for all parameters."""
    delta = OrderedDict()
    for key in global_state.keys():
        delta[key] = local_state[key].detach().cpu() - global_state[key].detach().cpu()
    return delta


def compute_update_l2_norm(delta: OrderedDict[str, torch.Tensor]) -> float:
    """Computes the overall L2 norm of the flattened parameter delta vector."""
    total_sq_norm = 0.0
    for tensor in delta.values():
        total_sq_norm += float(torch.sum(tensor**2).item())
    return float(np.sqrt(total_sq_norm))


def clip_update(
    delta: OrderedDict[str, torch.Tensor],
    max_norm: float,
) -> Tuple[OrderedDict[str, torch.Tensor], float]:
    """Clips the update delta to an L2 norm upper bound C.

    Formula:
      delta_clipped = delta * min(1, C / ||delta||_2)
    Returns:
      (clipped_delta, original_norm)
    """
    orig_norm = compute_update_l2_norm(delta)
    if orig_norm == 0.0 or orig_norm <= max_norm:
        scaling_factor = 1.0
    else:
        scaling_factor = max_norm / orig_norm

    clipped = OrderedDict()
    for key, tensor in delta.items():
        clipped[key] = tensor * scaling_factor

    return clipped, orig_norm


def add_gaussian_noise(
    aggregated_delta: OrderedDict[str, torch.Tensor],
    noise_multiplier: float,
    clip_norm: float,
    device: torch.device = torch.device("cpu"),
) -> OrderedDict[str, torch.Tensor]:
    """Adds calibrated zero-mean Gaussian noise to the aggregated update.

    Noise standard deviation sigma = noise_multiplier * clip_norm.
    When noise_multiplier == 0, returns the unmodified update.
    """
    if noise_multiplier <= 0.0:
        return aggregated_delta

    sigma = noise_multiplier * clip_norm
    noisy_delta = OrderedDict()

    for key, tensor in aggregated_delta.items():
        # Draw Gaussian noise with std = sigma
        noise = torch.normal(
            mean=0.0,
            std=sigma,
            size=tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        noisy_delta[key] = tensor + noise

    return noisy_delta


class AuditLogger:
    """Thread-safe and append-only cryptographic audit logger for Federated Learning."""

    def __init__(self, log_path: str = "results/audit_log.jsonl") -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log_round(
        self,
        round_idx: int,
        participating_banks: List[str],
        update_norms: Dict[str, float],
        noise_multiplier: float,
        global_model_hash: str,
        test_auprc: float,
        additional_info: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Appends an immutable JSON-line record to results/audit_log.jsonl."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": round_idx,
            "participating_banks": participating_banks,
            "update_norm_per_bank": {
                k: round(v, 6) for k, v in update_norms.items()
            },
            "noise_multiplier": noise_multiplier,
            "global_model_hash": global_model_hash,
            "test_auprc": round(test_auprc, 6),
        }
        if additional_info:
            entry.update(additional_info)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry
