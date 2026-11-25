from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from meddr.constants import DECISION_THRESHOLD, NORMALIZATION_EPSILON
from meddr.schemas import ResearchSample


def report_probabilities(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["name"]): float(row["adjusted_probability"])
        for row in report["medications"]
    }


def boundary_bce(sample: ResearchSample, report: dict[str, Any]) -> float:
    sample.require_training_labels()
    probabilities = report_probabilities(report)
    losses: list[float] = []
    for medication in sample.boundary_medications:
        probability = min(
            max(probabilities[medication.name], NORMALIZATION_EPSILON),
            1.0 - NORMALIZATION_EPSILON,
        )
        label = float(medication.label)
        losses.append(
            -(
                label * math.log(probability)
                + (1.0 - label) * math.log(1.0 - probability)
            )
        )
    return sum(losses) / len(losses)


def predicted_medications(
    sample: ResearchSample,
    report: dict[str, Any],
) -> set[int]:
    selected = set(sample.confident_accept_ids)
    probabilities = report_probabilities(report)
    for medication in sample.boundary_medications:
        if probabilities[medication.name] >= DECISION_THRESHOLD:
            selected.add(medication.identifier)
    return selected


def ddi_rate(medications: set[int], ddi_matrix: Any) -> float:
    identifiers = sorted(medications)
    pair_count = 0
    interaction_count = 0
    for index, first in enumerate(identifiers):
        for second in identifiers[index + 1 :]:
            pair_count += 1
            interaction_count += int(
                bool(ddi_matrix[first][second])
                or bool(ddi_matrix[second][first])
            )
    return interaction_count / pair_count if pair_count else 0.0


def correctness_by_medication(
    sample: ResearchSample,
    reports: list[dict[str, Any]],
) -> dict[str, float]:
    sample.require_training_labels()
    if not reports:
        raise ValueError("PNS estimation requires Reporter samples.")
    totals = {item.name: 0.0 for item in sample.boundary_medications}
    for report in reports:
        probabilities = report_probabilities(report)
        for medication in sample.boundary_medications:
            decision = int(
                probabilities[medication.name] >= DECISION_THRESHOLD
            )
            totals[medication.name] += float(decision == medication.label)
    return {
        name: total / len(reports)
        for name, total in totals.items()
    }


def pns_reward(
    sample: ResearchSample,
    full_reports: list[dict[str, Any]],
    counterfactual_reports: list[dict[str, Any]],
) -> float:
    diagnostics = pns_diagnostics(
        sample,
        full_reports,
        counterfactual_reports,
    )
    contributions = diagnostics["positive_contribution"]
    return sum(contributions.values()) / len(contributions)


def pns_diagnostics(
    sample: ResearchSample,
    full_reports: list[dict[str, Any]],
    counterfactual_reports: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return the medication-level terms used by Eq. (16)."""
    full = correctness_by_medication(sample, full_reports)
    counterfactual = correctness_by_medication(
        sample,
        counterfactual_reports,
    )
    signed_difference = {
        name: full[name] - counterfactual[name]
        for name in full
    }
    return {
        "full_correctness": full,
        "counterfactual_correctness": counterfactual,
        "signed_difference": signed_difference,
        "positive_contribution": {
            name: max(0.0, value)
            for name, value in signed_difference.items()
        },
    }


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    normalized = (
        array - float(array.mean())
    ) / (float(array.std()) + NORMALIZATION_EPSILON)
    return [float(value) for value in normalized]


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = torch.clamp(policy_logprobs - old_logprobs, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )
    per_action = -torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    )
    weights = sample_weights.to(
        device=per_action.device,
        dtype=per_action.dtype,
    )
    loss = torch.sum(per_action * weights)
    normalized_weights = weights / torch.clamp(
        weights.sum(),
        min=torch.finfo(weights.dtype).eps,
    )
    detached_log_ratio = log_ratio.detach()
    detached_ratio = ratio.detach()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mean_ratio": float(
            torch.sum(detached_ratio * normalized_weights).cpu()
        ),
        "clipped_ratio_rate": float(
            torch.sum(
                (torch.abs(ratio - 1.0) > clip_range).float()
                * normalized_weights
            ).cpu()
        ),
        "mean_advantage": float(
            torch.sum(advantages.detach() * normalized_weights).cpu()
        ),
        "approx_kl": float(
            torch.sum(
                (
                    (detached_ratio - 1.0)
                    - detached_log_ratio
                )
                * normalized_weights
            ).cpu()
        ),
        "mean_abs_log_ratio": float(
            torch.sum(
                torch.abs(detached_log_ratio) * normalized_weights
            ).cpu()
        ),
        "min_ratio": float(detached_ratio.min().cpu()),
        "max_ratio": float(detached_ratio.max().cpu()),
    }
