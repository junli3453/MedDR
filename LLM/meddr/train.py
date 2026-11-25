from __future__ import annotations

import argparse
import copy
from itertools import combinations
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from meddr.agents import Planner, Questioner, Reporter
from meddr.constants import (
    DECISION_THRESHOLD,
    GRPO_CLIP_RANGE,
    GRPO_GROUP_SIZE,
    GRPO_MAX_CLIPPED_RATIO_RATE,
    GRPO_MAX_GRAD_NORM,
    GRPO_TARGET_KL,
    MAX_RESEARCH_ROUNDS,
    PNS_INTERVENTION,
    PNS_REPORTER_SAMPLES,
    PNS_REWARD_WEIGHT,
    QUESTIONER_TEMPERATURE,
    QUESTIONER_TOP_P,
    REPORTER_MAX_CONDITIONAL_SHIFT,
)
from meddr.interventions import PNS_INTERVENTIONS
from meddr.io import append_jsonl, load_pickle, load_samples
from meddr.modeling import FrozenChatModelView, LocalChatModel, QuestionAction
from meddr.pipeline import MedDRPipeline, Rollout
from meddr.retrieval import Retriever
from meddr.rewards import ddi_rate, grpo_loss, normalize
from meddr.schemas import (
    PersonalizedPlan,
    ResearchSample,
    validate_and_align_ddi_constraints,
)


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description="Train the MedDR Questioner with GRPO."
    )
    data_inputs = arguments.add_argument_group("Data inputs")
    data_inputs.add_argument(
        "--train-data",
        default=os.getenv(
            "MEDDR_TRAIN_DATA",
            "data/LLM-data/train.jsonl",
        ),
    )
    data_inputs.add_argument(
        "--ddi-matrix",
        default=os.getenv(
            "MEDDR_DDI_MATRIX",
            "data/ready/ddi_A_final.pkl",
        ),
        help="Preprocessed DDI adjacency file aligned with the medication vocabulary.",
    )
    arguments.add_argument(
        "--questioner-model",
        default=os.getenv(
            "MEDDR_QUESTIONER_MODEL",
            "models/qwen3-8b",
        ),
    )
    arguments.add_argument(
        "--planner-model",
        default=os.getenv(
            "MEDDR_PLANNER_MODEL",
            "models/qwen3-8b",
        ),
    )
    arguments.add_argument(
        "--reporter-model",
        default=os.getenv(
            "MEDDR_REPORTER_MODEL",
            "models/qwen3-8b",
        ),
    )
    arguments.add_argument(
        "--retriever-model",
        default=os.getenv("MEDDR_RETRIEVER_MODEL"),
    )
    arguments.add_argument(
        "--retriever-base-url",
        default=os.getenv("MEDDR_RETRIEVER_ENDPOINT"),
    )
    arguments.add_argument(
        "--retriever-timeout",
        type=float,
        default=float(os.getenv("MEDDR_RETRIEVER_TIMEOUT", "120")),
    )
    arguments.add_argument(
        "--retriever-max-retries",
        type=int,
        default=int(os.getenv("MEDDR_RETRIEVER_MAX_RETRIES", "3")),
    )
    arguments.add_argument(
        "--retriever-thinking",
        choices=("enabled", "disabled"),
        default=os.getenv("MEDDR_RETRIEVER_THINKING") or None,
    )
    arguments.add_argument("--adapter-path")
    arguments.add_argument("--output-dir", default="outputs/meddr")
    arguments.add_argument(
        "--device-map",
        default=os.getenv("MEDDR_DEVICE_MAP", "balanced"),
        help=(
            "Transformers device map. The balanced default shards the shared "
            "local model across all visible GPUs."
        ),
    )
    arguments.add_argument(
        "--model-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=os.getenv("MEDDR_MODEL_DTYPE", "float16"),
        help=(
            "Local-model forward dtype. float16 provides finer policy-ratio "
            "resolution than bfloat16 at the same parameter memory cost."
        ),
    )
    arguments.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("MEDDR_LEARNING_RATE", "5e-8")),
    )
    arguments.add_argument("--max-updates", type=int, default=1)
    arguments.add_argument("--save-every", type=int, default=10)
    arguments.add_argument(
        "--group-size",
        type=int,
        default=int(
            os.getenv("MEDDR_GRPO_GROUP_SIZE", str(GRPO_GROUP_SIZE))
        ),
    )
    arguments.add_argument(
        "--sample-start-index",
        type=int,
        default=int(os.getenv("MEDDR_SAMPLE_START_INDEX", "0")),
        help=(
            "Zero-based dataset offset for the first patient batch. "
            "Subsequent updates continue cyclically from this offset."
        ),
    )
    arguments.add_argument(
        "--pns-samples",
        type=int,
        default=int(
            os.getenv(
                "MEDDR_PNS_REPORTER_SAMPLES",
                str(PNS_REPORTER_SAMPLES),
            )
        ),
    )
    arguments.add_argument(
        "--pns-intervention",
        choices=PNS_INTERVENTIONS,
        default=os.getenv(
            "MEDDR_PNS_INTERVENTION",
            PNS_INTERVENTION,
        ),
        help=(
            "PNS counterfactual: evidence removal, removal of personalized "
            "information (rm), or another-patient toxic objective (tox)."
        ),
    )
    arguments.add_argument(
        "--optimization-micro-batch",
        type=int,
        default=int(os.getenv("MEDDR_OPTIMIZATION_MICRO_BATCH", "8")),
    )
    arguments.add_argument(
        "--optimization-epochs",
        type=int,
        default=int(os.getenv("MEDDR_GRPO_OPTIMIZATION_EPOCHS", "1")),
        help=(
            "Optimizer passes over each fixed rollout batch. Values above one "
            "allow the current/old policy ratio and GRPO clipping to become "
            "active."
        ),
    )
    arguments.add_argument(
        "--max-grad-norm",
        type=float,
        default=float(
            os.getenv(
                "MEDDR_MAX_GRAD_NORM",
                str(GRPO_MAX_GRAD_NORM),
            )
        ),
    )
    arguments.add_argument(
        "--target-kl",
        type=float,
        default=float(
            os.getenv("MEDDR_GRPO_TARGET_KL", str(GRPO_TARGET_KL))
        ),
    )
    arguments.add_argument(
        "--max-clipped-ratio-rate",
        type=float,
        default=float(
            os.getenv(
                "MEDDR_GRPO_MAX_CLIPPED_RATIO_RATE",
                str(GRPO_MAX_CLIPPED_RATIO_RATE),
            )
        ),
    )
    arguments.add_argument(
        "--reporter-max-conditional-shift",
        type=float,
        default=float(
            os.getenv(
                "MEDDR_REPORTER_MAX_CONDITIONAL_SHIFT",
                str(REPORTER_MAX_CONDITIONAL_SHIFT),
            )
        ),
        help=(
            "Maximum absolute Reporter probability shift when retrieved "
            "evidence explicitly remains conditional."
        ),
    )
    arguments.add_argument("--questioner-max-tokens", type=int, default=256)
    arguments.add_argument("--planner-max-tokens", type=int, default=768)
    arguments.add_argument("--reporter-max-tokens", type=int, default=1024)
    arguments.add_argument("--retriever-max-tokens", type=int, default=1024)
    arguments.add_argument("--seed", type=int, default=42)
    return arguments


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_updates < 1:
        raise ValueError("--max-updates must be positive.")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive.")
    if args.optimization_micro_batch < 1:
        raise ValueError("--optimization-micro-batch must be positive.")
    if args.optimization_epochs < 1:
        raise ValueError("--optimization-epochs must be positive.")
    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2.")
    if args.sample_start_index < 0:
        raise ValueError("--sample-start-index cannot be negative.")
    if args.pns_samples < 1:
        raise ValueError("--pns-samples must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive.")
    if args.target_kl <= 0.0:
        raise ValueError("--target-kl must be positive.")
    if not 0.0 <= args.max_clipped_ratio_rate <= 1.0:
        raise ValueError(
            "--max-clipped-ratio-rate must be within [0, 1]."
        )
    if not 0.0 < args.reporter_max_conditional_shift <= 1.0:
        raise ValueError(
            "--reporter-max-conditional-shift must be in (0, 1]."
        )
    if args.retriever_timeout <= 0.0:
        raise ValueError("--retriever-timeout must be positive.")
    if args.retriever_max_retries < 0:
        raise ValueError("--retriever-max-retries cannot be negative.")


def attach_advantages(
    rollouts: list[Rollout],
    group_size: int,
) -> None:
    if len(rollouts) != group_size:
        raise ValueError(
            f"Each patient requires exactly {group_size} rollouts."
        )
    global_advantages = normalize(
        [rollout.global_reward for rollout in rollouts]
    )
    step_count = len(rollouts[0].steps)
    if any(len(rollout.steps) != step_count for rollout in rollouts):
        raise ValueError("All rollouts for one patient must have equal length.")
    for step_index in range(step_count):
        process_advantages = normalize(
            [
                rollout.steps[step_index].pns_reward
                for rollout in rollouts
            ]
        )
        for rollout_index, rollout in enumerate(rollouts):
            rollout.steps[step_index].advantage = (
                global_advantages[rollout_index]
                + PNS_REWARD_WEIGHT * process_advantages[rollout_index]
            )


def flatten_training_rows(
    patient_groups: list[list[Rollout]],
) -> list[tuple[QuestionAction, float, float]]:
    patient_count = len(patient_groups)
    rows: list[tuple[QuestionAction, float, float]] = []
    for rollouts in patient_groups:
        group_size = len(rollouts)
        if group_size < 2:
            raise ValueError("Every patient group requires at least two rollouts.")
        for rollout in rollouts:
            action_count = len(rollout.steps)
            weight = 1.0 / (
                patient_count * group_size * action_count
            )
            for step in rollout.steps:
                rows.append((step.action, step.advantage, weight))
    return rows


def chunks(
    values: list[tuple[QuestionAction, float, float]],
    size: int,
) -> Iterable[list[tuple[QuestionAction, float, float]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def evaluate_questioner_policy(
    questioner: Questioner,
    rows: list[tuple[QuestionAction, float, float]],
    micro_batch: int,
) -> dict[str, float]:
    questioner.model.model.eval()
    totals = {
        "loss": 0.0,
        "mean_ratio": 0.0,
        "clipped_ratio_rate": 0.0,
        "mean_advantage": 0.0,
        "approx_kl": 0.0,
        "mean_abs_log_ratio": 0.0,
        "min_ratio": float("inf"),
        "max_ratio": 0.0,
    }
    for batch in chunks(rows, micro_batch):
        actions = [row[0] for row in batch]
        policy_logprobs = questioner.model.logprobs(
            [action.prompt for action in actions],
            [action.completion for action in actions],
        )
        policy_device = policy_logprobs.device
        advantages = torch.tensor(
            [row[1] for row in batch],
            dtype=torch.float32,
            device=policy_device,
        )
        weights = torch.tensor(
            [row[2] for row in batch],
            dtype=torch.float32,
            device=policy_device,
        )
        old_logprobs = torch.tensor(
            [action.old_logprob for action in actions],
            dtype=torch.float32,
            device=policy_device,
        )
        _, statistics = grpo_loss(
            policy_logprobs,
            old_logprobs,
            advantages,
            weights,
            clip_range=GRPO_CLIP_RANGE,
        )
        chunk_weight = float(weights.sum().detach().cpu())
        totals["loss"] += statistics["loss"]
        for key in (
            "mean_ratio",
            "clipped_ratio_rate",
            "mean_advantage",
            "approx_kl",
            "mean_abs_log_ratio",
        ):
            totals[key] += statistics[key] * chunk_weight
        totals["min_ratio"] = min(
            totals["min_ratio"],
            statistics["min_ratio"],
        )
        totals["max_ratio"] = max(
            totals["max_ratio"],
            statistics["max_ratio"],
        )
    questioner.model.model.train()
    return totals


def optimize_questioner(
    questioner: Questioner,
    patient_groups: list[list[Rollout]],
    optimizer: torch.optim.Optimizer,
    micro_batch: int,
    optimization_epochs: int,
    max_grad_norm: float,
    target_kl: float,
    max_clipped_ratio_rate: float,
) -> dict[str, float | int]:
    if optimization_epochs < 1:
        raise ValueError("optimization_epochs must be positive.")
    rows = flatten_training_rows(patient_groups)
    questioner.model.model.train()
    epoch_statistics: list[dict[str, float]] = []
    early_stop_statistics: dict[str, float] | None = None
    applied_epochs = 0
    for epoch in range(1, optimization_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "mean_ratio": 0.0,
            "clipped_ratio_rate": 0.0,
            "mean_advantage": 0.0,
            "approx_kl": 0.0,
            "mean_abs_log_ratio": 0.0,
            "min_ratio": float("inf"),
            "max_ratio": 0.0,
        }
        for batch in chunks(rows, micro_batch):
            actions = [row[0] for row in batch]
            policy_logprobs = questioner.model.logprobs(
                [action.prompt for action in actions],
                [action.completion for action in actions],
            )
            policy_device = policy_logprobs.device
            advantages = torch.tensor(
                [row[1] for row in batch],
                dtype=torch.float32,
                device=policy_device,
            )
            weights = torch.tensor(
                [row[2] for row in batch],
                dtype=torch.float32,
                device=policy_device,
            )
            old_logprobs = torch.tensor(
                [action.old_logprob for action in actions],
                dtype=torch.float32,
                device=policy_device,
            )
            loss, statistics = grpo_loss(
                policy_logprobs,
                old_logprobs,
                advantages,
                weights,
                clip_range=GRPO_CLIP_RANGE,
            )
            loss.backward()
            chunk_weight = float(weights.sum().detach().cpu())
            totals["loss"] += statistics["loss"]
            for key in (
                "mean_ratio",
                "clipped_ratio_rate",
                "mean_advantage",
                "approx_kl",
                "mean_abs_log_ratio",
            ):
                totals[key] += statistics[key] * chunk_weight
            totals["min_ratio"] = min(
                totals["min_ratio"],
                statistics["min_ratio"],
            )
            totals["max_ratio"] = max(
                totals["max_ratio"],
                statistics["max_ratio"],
            )
        if (
            epoch > 1
            and (
                totals["approx_kl"] > target_kl
                or totals["clipped_ratio_rate"]
                > max_clipped_ratio_rate
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            early_stop_statistics = dict(totals)
            print(
                f"optimization_epoch={epoch}/{optimization_epochs} "
                "optimizer_step=skipped "
                f"mean_ratio={totals['mean_ratio']:.6f} "
                f"ratio_range=[{totals['min_ratio']:.6f},"
                f"{totals['max_ratio']:.6f}] "
                f"clipped_ratio_rate="
                f"{totals['clipped_ratio_rate']:.6f} "
                f"approx_kl={totals['approx_kl']:.6f} "
                "reason=trust_region_limit",
                flush=True,
            )
            break
        parameters_with_gradients = [
            parameter
            for parameter in questioner.model.model.parameters()
            if parameter.grad is not None
        ]
        pre_clip_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters_with_gradients,
                max_grad_norm,
            ).detach().cpu()
        )
        post_clip_grad_norm = min(
            pre_clip_grad_norm,
            max_grad_norm,
        )
        totals["grad_norm"] = pre_clip_grad_norm
        totals["post_clip_grad_norm"] = post_clip_grad_norm
        totals["gradient_was_clipped"] = float(
            pre_clip_grad_norm > max_grad_norm
        )
        parameters_and_backups = [
            (parameter, parameter.detach().clone())
            for parameter in parameters_with_gradients
        ]
        optimizer_backup = copy.deepcopy(optimizer.state_dict())
        initial_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        post_update_statistics: dict[str, float] | None = None
        accepted = False
        no_op_detected = False
        backtracking_steps = 0
        backtracking_factor = 0.5
        minimum_policy_movement = 1e-7
        max_backtracking_attempts = 8
        last_nonzero_attempt: int | None = None
        last_nonzero_statistics: dict[str, float] | None = None
        for attempt in range(max_backtracking_attempts):
            if attempt:
                for parameter, backup in parameters_and_backups:
                    parameter.data.copy_(backup)
                optimizer.load_state_dict(
                    copy.deepcopy(optimizer_backup)
                )
                for group, learning_rate in zip(
                    optimizer.param_groups,
                    initial_learning_rates,
                ):
                    group["lr"] = learning_rate * (
                        backtracking_factor ** attempt
                    )
            optimizer.step()
            post_update_statistics = evaluate_questioner_policy(
                questioner,
                rows,
                micro_batch,
            )
            no_op_detected = (
                post_update_statistics["mean_abs_log_ratio"]
                <= minimum_policy_movement
            )
            if not no_op_detected:
                last_nonzero_attempt = attempt
                last_nonzero_statistics = dict(
                    post_update_statistics
                )
            print(
                f"optimization_epoch={epoch}/{optimization_epochs} "
                f"backtracking_probe={attempt} "
                f"learning_rate="
                f"{optimizer.param_groups[0]['lr']:.3e} "
                f"mean_abs_log_ratio="
                f"{post_update_statistics['mean_abs_log_ratio']:.6f} "
                f"clipped_ratio_rate="
                f"{post_update_statistics['clipped_ratio_rate']:.6f} "
                f"approx_kl="
                f"{post_update_statistics['approx_kl']:.6f} "
                f"no_op={int(no_op_detected)}",
                flush=True,
            )
            if (
                post_update_statistics["approx_kl"] <= target_kl
                and post_update_statistics["clipped_ratio_rate"]
                <= max_clipped_ratio_rate
                and not no_op_detected
            ):
                accepted = True
                backtracking_steps = attempt
                break
            if no_op_detected:
                backtracking_steps = attempt
                break
        trust_region_relaxed = False
        precision_boundary_detected = (
            no_op_detected
            and last_nonzero_attempt is not None
            and last_nonzero_statistics is not None
        )
        relaxed_max_kl = max(0.1, 10.0 * target_kl)
        relaxed_max_clipped_ratio_rate = max(
            0.75,
            max_clipped_ratio_rate,
        )
        relaxed_update_is_safe = bool(
            last_nonzero_statistics is not None
            and np.isfinite(
                list(last_nonzero_statistics.values())
            ).all()
            and last_nonzero_statistics["approx_kl"]
            <= relaxed_max_kl
            and last_nonzero_statistics["clipped_ratio_rate"]
            <= relaxed_max_clipped_ratio_rate
            and last_nonzero_statistics["min_ratio"] >= 0.25
            and last_nonzero_statistics["max_ratio"] <= 4.0
        )
        if (
            not accepted
            and last_nonzero_attempt is not None
            and last_nonzero_statistics is not None
            and relaxed_update_is_safe
        ):
            if (
                last_nonzero_attempt is None
                or last_nonzero_statistics is None
            ):
                raise AssertionError(
                    "Precision-boundary candidate is missing."
                )
            for parameter, backup in parameters_and_backups:
                parameter.data.copy_(backup)
            optimizer.load_state_dict(
                copy.deepcopy(optimizer_backup)
            )
            for group, learning_rate in zip(
                optimizer.param_groups,
                initial_learning_rates,
            ):
                group["lr"] = learning_rate * (
                    backtracking_factor ** last_nonzero_attempt
                )
            optimizer.step()
            post_update_statistics = last_nonzero_statistics
            accepted = True
            no_op_detected = False
            trust_region_relaxed = True
            backtracking_steps = last_nonzero_attempt
            print(
                f"optimization_epoch={epoch}/{optimization_epochs} "
                "optimizer_step=minimum_measurable_update "
                f"backtracking_steps={backtracking_steps} "
                f"learning_rate="
                f"{optimizer.param_groups[0]['lr']:.3e} "
                "reason="
                + (
                    "numerical_precision_boundary"
                    if precision_boundary_detected
                    else "bounded_backtracking_fallback"
                ),
                flush=True,
            )
        if not accepted:
            for parameter, backup in parameters_and_backups:
                parameter.data.copy_(backup)
            optimizer.load_state_dict(optimizer_backup)
            for group, learning_rate in zip(
                optimizer.param_groups,
                initial_learning_rates,
            ):
                if not no_op_detected:
                    group["lr"] = learning_rate * (
                        backtracking_factor
                        ** max_backtracking_attempts
                    )
            optimizer.zero_grad(set_to_none=True)
            failed_post_update = post_update_statistics
            early_stop_statistics = dict(totals)
            if post_update_statistics is not None:
                for key in (
                    "loss",
                    "mean_ratio",
                    "clipped_ratio_rate",
                    "mean_advantage",
                    "approx_kl",
                    "mean_abs_log_ratio",
                    "min_ratio",
                    "max_ratio",
                ):
                    totals[f"pre_update_{key}"] = totals[key]
                for key in (
                    "loss",
                    "mean_ratio",
                    "clipped_ratio_rate",
                    "mean_advantage",
                    "approx_kl",
                    "mean_abs_log_ratio",
                    "min_ratio",
                    "max_ratio",
                ):
                    totals[f"rollback_probe_{key}"] = (
                        failed_post_update[key]
                    )
            totals["optimizer_backtracking_steps"] = float(
                backtracking_steps
                if no_op_detected
                else max_backtracking_attempts
            )
            totals["effective_learning_rate"] = 0.0
            totals["next_learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )
            totals["optimizer_step_applied"] = 0.0
            totals["optimizer_noop_detected"] = float(no_op_detected)
            totals["optimizer_trust_region_relaxed"] = 0.0
            totals["optimizer_precision_boundary_detected"] = float(
                precision_boundary_detected
            )
            epoch_statistics.append(totals)
            print(
                f"optimization_epoch={epoch}/{optimization_epochs} "
                "optimizer_step=rolled_back "
                "reason="
                + (
                    "no_measurable_policy_movement"
                    if no_op_detected
                    else "trust_region_backtracking_exhausted"
                ),
                flush=True,
            )
            break
        if post_update_statistics is None:
            raise AssertionError("Post-update policy statistics are missing.")
        for key in (
            "loss",
            "mean_ratio",
            "clipped_ratio_rate",
            "mean_advantage",
            "approx_kl",
            "mean_abs_log_ratio",
            "min_ratio",
            "max_ratio",
        ):
            totals[f"pre_update_{key}"] = totals[key]
            totals[key] = post_update_statistics[key]
        totals["optimizer_backtracking_steps"] = float(
            backtracking_steps
        )
        totals["effective_learning_rate"] = float(
            optimizer.param_groups[0]["lr"]
        )
        totals["next_learning_rate"] = float(
            optimizer.param_groups[0]["lr"]
        )
        totals["optimizer_step_applied"] = 1.0
        totals["optimizer_noop_detected"] = 0.0
        totals["optimizer_trust_region_relaxed"] = float(
            trust_region_relaxed
        )
        totals["optimizer_relaxed_max_kl"] = relaxed_max_kl
        totals["optimizer_relaxed_max_clipped_ratio_rate"] = (
            relaxed_max_clipped_ratio_rate
        )
        totals["optimizer_precision_boundary_detected"] = float(
            precision_boundary_detected
        )
        applied_epochs += 1
        epoch_statistics.append(totals)
        print(
            f"optimization_epoch={epoch}/{optimization_epochs} "
            f"loss={totals['loss']:.6f} "
            f"mean_ratio={totals['mean_ratio']:.6f} "
            f"ratio_range=[{totals['min_ratio']:.6f},"
            f"{totals['max_ratio']:.6f}] "
            f"clipped_ratio_rate={totals['clipped_ratio_rate']:.6f} "
            f"approx_kl={totals['approx_kl']:.6f} "
            f"grad_norm={totals['grad_norm']:.6f} "
            f"post_clip_grad_norm="
            f"{totals['post_clip_grad_norm']:.6f} "
            f"backtracking_steps={backtracking_steps} "
            f"effective_learning_rate="
            f"{totals['effective_learning_rate']:.3e}",
            flush=True,
        )
    questioner.model.model.eval()
    first = epoch_statistics[0]
    final = dict(epoch_statistics[-1])
    if early_stop_statistics is not None:
        for key in (
            "loss",
            "mean_ratio",
            "clipped_ratio_rate",
            "mean_advantage",
            "approx_kl",
            "mean_abs_log_ratio",
            "min_ratio",
            "max_ratio",
        ):
            final[key] = early_stop_statistics[key]
    return {
        **final,
        "optimization_epochs": optimization_epochs,
        "optimization_epochs_applied": applied_epochs,
        "optimization_early_stop": early_stop_statistics is not None,
        "first_epoch_loss": first["loss"],
        "first_epoch_mean_ratio": first["mean_ratio"],
        "first_epoch_pre_update_mean_ratio": first[
            "pre_update_mean_ratio"
        ],
        "first_epoch_clipped_ratio_rate": (
            first["clipped_ratio_rate"]
        ),
        "mean_grad_norm": mean(
            [item["grad_norm"] for item in epoch_statistics]
        ),
        "mean_post_clip_grad_norm": mean(
            [
                item["post_clip_grad_norm"]
                for item in epoch_statistics
            ]
        ),
        "gradient_clip_rate": mean(
            [
                item["gradient_was_clipped"]
                for item in epoch_statistics
            ]
        ),
        "max_grad_norm": max_grad_norm,
        "optimizer_backtracking_steps": sum(
            item.get("optimizer_backtracking_steps", 0.0)
            for item in epoch_statistics
        ),
        "optimizer_noop_detected": max(
            item.get("optimizer_noop_detected", 0.0)
            for item in epoch_statistics
        ),
        "optimizer_trust_region_relaxed": max(
            item.get("optimizer_trust_region_relaxed", 0.0)
            for item in epoch_statistics
        ),
        "optimizer_precision_boundary_detected": max(
            item.get("optimizer_precision_boundary_detected", 0.0)
            for item in epoch_statistics
        ),
        "effective_learning_rate": float(
            final.get("effective_learning_rate", 0.0)
        ),
        "next_learning_rate": float(
            optimizer.param_groups[0]["lr"]
        ),
    }


def patient_batch(
    samples: list[ResearchSample],
    update: int,
    sample_start_index: int,
) -> list[ResearchSample]:
    index = (sample_start_index + update - 1) % len(samples)
    return [samples[index]]


def consensus_recommendation(
    sample: ResearchSample,
    rollouts: list[Rollout],
    ddi_matrix: object,
    *,
    update: int,
    patient_index: int,
) -> dict[str, object]:
    if not rollouts:
        raise ValueError(
            "A consensus recommendation requires at least one rollout."
        )
    boundary_rows: list[dict[str, object]] = []
    selected_ids = set(sample.confident_accept_ids)
    selected_names = {
        medication.name for medication in sample.confident_accept
    }
    for medication in sample.boundary_medications:
        adjusted_probabilities = [
            float(
                next(
                    row["adjusted_probability"]
                    for row in rollout.final_report["medications"]
                    if row["name"] == medication.name
                )
            )
            for rollout in rollouts
        ]
        mean_probability = mean(adjusted_probabilities)
        recommended = mean_probability >= DECISION_THRESHOLD
        if recommended:
            selected_ids.add(medication.identifier)
            selected_names.add(medication.name)
        boundary_rows.append(
            {
                "name": medication.name,
                "base_probability": medication.probability,
                "mean_adjusted_probability": mean_probability,
                "min_adjusted_probability": min(
                    adjusted_probabilities
                ),
                "max_adjusted_probability": max(
                    adjusted_probabilities
                ),
                "recommended": recommended,
            }
        )
    return {
        "update": update,
        "patient_index": patient_index,
        "sample_id": sample.sample_id,
        "consensus_method": (
            "mean_adjusted_probability_across_rollouts"
        ),
        "rollout_count": len(rollouts),
        "selection_threshold": DECISION_THRESHOLD,
        "boundary_medications": boundary_rows,
        "confidently_included_medications": sorted(
            medication.name
            for medication in sample.confident_accept
        ),
        "recommended_medications": sorted(selected_names),
        "consensus_ddi_rate": ddi_rate(
            selected_ids,
            ddi_matrix,
        ),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def planner_distribution_diagnostics(
    sample_ids: Iterable[str],
    samples: list[ResearchSample],
    plan_cache: dict[str, PersonalizedPlan],
    generation_cache: dict[str, dict[str, object] | None],
) -> dict[str, object]:
    """Summarize genuine Planner multiplicity without counting fallbacks."""
    sample_lookup = {sample.sample_id: sample for sample in samples}
    eligible_ids = list(
        dict.fromkeys(
            sample_id
            for sample_id in sample_ids
            if sample_id in plan_cache
            and sample_id in sample_lookup
            and len(sample_lookup[sample_id].boundary_medications) > 1
        )
    )
    primary_counts: list[int] = []
    final_counts: list[int] = []
    fallback_values: list[float] = []
    review_requested_values: list[float] = []
    review_selected_values: list[float] = []
    review_attempt_counts: list[float] = []
    review_unresolved_values: list[float] = []
    component_counts: list[int] = []
    reconstruction_requested_values: list[float] = []
    reconstruction_selected_values: list[float] = []
    reconstruction_failed_values: list[float] = []
    for sample_id in eligible_ids:
        state = generation_cache.get(sample_id) or {}
        primary_counts.append(int(state.get("primary_objective_count", 0)))
        final_counts.append(
            len(plan_cache[sample_id].research_objectives)
        )
        fallback_values.append(float(bool(state.get("fallback_used", False))))
        review_requested_values.append(
            float(bool(state.get("structural_review_requested", False)))
        )
        review_selected_values.append(
            float(bool(state.get("structural_review_selected", False)))
        )
        review_attempt_counts.append(
            float(state.get("structural_review_attempt_count", 0))
        )
        review_unresolved_values.append(
            float(bool(state.get("structural_review_unresolved", False)))
        )
        component_counts.append(
            int(state.get("source_grounded_component_count", 0))
        )
        reconstruction_requested_values.append(float(bool(
            state.get("source_grounded_reconstruction_requested", False)
        )))
        reconstruction_selected_values.append(float(bool(
            state.get("source_grounded_reconstruction_selected", False)
        )))
        reconstruction_failed_values.append(float(bool(
            state.get("source_grounded_reconstruction_failed", False)
        )))

    def rate(values: list[float]) -> float:
        return mean(values) if values else 0.0

    def multi_rate(counts: list[int]) -> float:
        return rate([float(count > 1) for count in counts])

    def distribution(counts: list[int]) -> dict[str, int]:
        return {
            str(count): counts.count(count)
            for count in sorted(set(counts))
        }

    return {
        "eligible_multi_boundary_sample_count": len(eligible_ids),
        "primary_objective_count_mean": (
            mean([float(count) for count in primary_counts])
            if primary_counts
            else 0.0
        ),
        "primary_objective_count_distribution": distribution(
            primary_counts
        ),
        "primary_multi_objective_rate": multi_rate(primary_counts),
        "final_objective_count_mean": (
            mean([float(count) for count in final_counts])
            if final_counts
            else 0.0
        ),
        "final_objective_count_distribution": distribution(final_counts),
        "final_multi_objective_rate": multi_rate(final_counts),
        "fallback_rate": rate(fallback_values),
        "source_grounded_component_count_mean": (
            mean([float(count) for count in component_counts])
            if component_counts
            else 0.0
        ),
        "source_grounded_component_count_distribution": distribution(
            component_counts
        ),
        "source_grounded_multi_component_rate": multi_rate(
            component_counts
        ),
        "source_grounded_reconstruction_requested_rate": rate(
            reconstruction_requested_values
        ),
        "source_grounded_reconstruction_selected_rate": rate(
            reconstruction_selected_values
        ),
        "source_grounded_reconstruction_failed_rate": rate(
            reconstruction_failed_values
        ),
        "structural_review_requested_rate": rate(
            review_requested_values
        ),
        "structural_review_selected_rate": rate(
            review_selected_values
        ),
        "structural_review_attempt_count_mean": rate(
            review_attempt_counts
        ),
        "structural_review_unresolved_rate": rate(
            review_unresolved_values
        ),
    }


QUESTION_DIAGNOSTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "both",
    "compared",
    "does",
    "in",
    "influence",
    "of",
    "patient",
    "patients",
    "presence",
    "the",
    "this",
    "to",
    "with",
    "without",
}


def diagnostic_question_tokens(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", question.casefold())
        if token not in QUESTION_DIAGNOSTIC_STOPWORDS
    }


def within_objective_question_jaccards(
    steps: list[RolloutStep],
) -> list[float]:
    grouped: dict[str, list[set[str]]] = {}
    for step in steps:
        grouped.setdefault(step.action.objective_id, []).append(
            diagnostic_question_tokens(step.action.question)
        )
    similarities: list[float] = []
    for questions in grouped.values():
        for first, second in combinations(questions, 2):
            union = first | second
            similarities.append(
                len(first & second) / len(union) if union else 1.0
            )
    return similarities


def mean_within_objective_question_jaccard(
    steps: list[RolloutStep],
) -> float:
    similarities = within_objective_question_jaccards(steps)
    return mean(similarities) if similarities else 0.0


def pns_by_step(
    rollouts: list[Rollout],
) -> dict[str, dict[str, object]]:
    grouped: dict[int, list[float]] = {}
    medication_terms: dict[
        int,
        dict[str, dict[str, list[float]]],
    ] = {}
    for rollout in rollouts:
        for step in rollout.steps:
            step_number = step.evidence.step
            grouped.setdefault(step_number, []).append(
                step.pns_reward
            )
            for medication, signed in (
                step.signed_pns_difference or {}
            ).items():
                terms = medication_terms.setdefault(
                    step_number,
                    {},
                ).setdefault(
                    medication,
                    {
                        "full": [],
                        "counterfactual": [],
                        "signed": [],
                        "positive": [],
                    },
                )
                terms["full"].append(
                    (step.full_correctness or {})[medication]
                )
                terms["counterfactual"].append(
                    (step.counterfactual_correctness or {})[medication]
                )
                terms["signed"].append(signed)
                terms["positive"].append(
                    (step.positive_pns_contribution or {})[medication]
                )
    output: dict[str, dict[str, object]] = {}
    for step, values in sorted(grouped.items()):
        output[str(step)] = {
            "action_count": len(values),
            "mean": mean(values),
            "std": float(np.std(values)),
            "min": min(values),
            "max": max(values),
            "positive_rate": mean(
                [float(value > 0.0) for value in values]
            ),
            "medications": {
                medication: {
                    "full_correctness_mean": mean(terms["full"]),
                    "counterfactual_correctness_mean": mean(
                        terms["counterfactual"]
                    ),
                    "signed_difference_mean": mean(terms["signed"]),
                    "positive_contribution_mean": mean(
                        terms["positive"]
                    ),
                    "positive_contribution_rate": mean(
                        [
                            float(value > 0.0)
                            for value in terms["positive"]
                        ]
                    ),
                    "monotonicity_violation_rate": mean(
                        [
                            float(value < -1e-12)
                            for value in terms["signed"]
                        ]
                    ),
                }
                for medication, terms in sorted(
                    medication_terms.get(step, {}).items()
                )
            },
        }
    return output


def base_decision_diagnostics(
    samples: list[ResearchSample],
    ddi_matrix: object,
) -> dict[str, float | int]:
    correct: list[float] = []
    safety: list[float] = []
    for sample in samples:
        selected = set(sample.confident_accept_ids)
        for medication in sample.boundary_medications:
            decision = int(
                medication.probability >= DECISION_THRESHOLD
            )
            correct.append(float(decision == medication.label))
            if decision:
                selected.add(medication.identifier)
        safety.append(ddi_rate(selected, ddi_matrix))
    return {
        "base_boundary_accuracy": mean(correct),
        "base_incorrect_boundary_count": sum(
            value == 0.0 for value in correct
        ),
        "base_ddi_rate": mean(safety),
    }


def main() -> None:
    args = parser().parse_args()
    validate_arguments(args)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    recommendations_path = output_dir / "recommendations.jsonl"
    existing_training_outputs = [
        path
        for path in (
            metrics_path,
            recommendations_path,
        )
        if path.exists()
    ]
    if existing_training_outputs:
        raise FileExistsError(
            "The output directory already contains training records: "
            + ", ".join(path.name for path in existing_training_outputs)
            + ". Use a new --output-dir to keep runs independent."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_samples(args.train_data)
    if not samples:
        raise ValueError("No training samples were loaded.")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique.")
    for sample in samples:
        sample.require_training_labels()
    ddi_matrix = load_pickle(args.ddi_matrix)
    validate_and_align_ddi_constraints(samples, ddi_matrix)

    questioner_model = LocalChatModel(
        args.questioner_model,
        device_map=args.device_map,
        trainable=True,
        adapter_path=args.adapter_path,
        torch_dtype=args.model_dtype,
    )
    questioner_base_view = FrozenChatModelView(questioner_model)
    planner_model = (
        questioner_base_view
        if args.planner_model == args.questioner_model
        else LocalChatModel(
            args.planner_model,
            device_map=args.device_map,
            trainable=False,
            torch_dtype=args.model_dtype,
        )
    )
    planner = Planner(
        planner_model,
        max_new_tokens=args.planner_max_tokens,
    )
    questioner = Questioner(
        questioner_model,
        temperature=QUESTIONER_TEMPERATURE,
        top_p=QUESTIONER_TOP_P,
        max_new_tokens=args.questioner_max_tokens,
    )
    if args.reporter_model == args.questioner_model:
        reporter_model = questioner_base_view
    elif args.reporter_model == args.planner_model:
        reporter_model = planner_model
    else:
        reporter_model = LocalChatModel(
            args.reporter_model,
            device_map=args.device_map,
            trainable=False,
            torch_dtype=args.model_dtype,
        )
    reporter = Reporter(
        reporter_model,
        max_new_tokens=args.reporter_max_tokens,
        max_conditional_shift=(
            args.reporter_max_conditional_shift
        ),
    )
    retriever = Retriever(
        args.retriever_model,
        base_url=args.retriever_base_url,
        max_tokens=args.retriever_max_tokens,
        timeout=args.retriever_timeout,
        max_retries=args.retriever_max_retries,
        thinking=args.retriever_thinking,
    )
    pipeline = MedDRPipeline(
        planner,
        questioner,
        retriever,
        reporter,
        ddi_matrix=ddi_matrix,
        rounds=MAX_RESEARCH_ROUNDS,
        pns_samples=args.pns_samples,
        pns_intervention=args.pns_intervention,
        intervention_samples=tuple(samples),
    )
    trainable_parameters = [
        parameter
        for parameter in questioner_model.model.parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in trainable_parameters
        }
    )
    print(
        f"model_dtype={args.model_dtype} "
        "trainable_parameter_dtypes="
        + ",".join(trainable_parameter_dtypes),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )
    plan_cache: dict[str, PersonalizedPlan] = {}
    plan_generation_cache: dict[str, dict[str, object] | None] = {}

    for update in range(1, args.max_updates + 1):
        groups: list[list[Rollout]] = []
        update_samples = patient_batch(
            samples,
            update,
            args.sample_start_index,
        )
        base_diagnostics = base_decision_diagnostics(
            update_samples,
            ddi_matrix,
        )
        total_rollouts = len(update_samples) * args.group_size
        completed_rollouts = 0
        sample_ids = ",".join(
            sample.sample_id for sample in update_samples
        )
        print(
            f"update={update} rollout_collection_started "
            f"patients={len(update_samples)} "
            f"sample_ids={sample_ids} "
            f"group_size={args.group_size} "
            f"pns_samples={args.pns_samples} "
            f"pns_intervention={args.pns_intervention} "
            f"learning_rate={args.learning_rate} "
            f"reporter_max_conditional_shift="
            f"{args.reporter_max_conditional_shift} "
            f"base_boundary_accuracy="
            f"{base_diagnostics['base_boundary_accuracy']:.6f} "
            f"base_incorrect_boundary_count="
            f"{base_diagnostics['base_incorrect_boundary_count']} "
            f"base_ddi_rate={base_diagnostics['base_ddi_rate']:.6f} "
            f"total_rollouts={total_rollouts}",
            flush=True,
        )
        if (
            args.pns_intervention == "removal"
            and base_diagnostics["base_incorrect_boundary_count"] == 0
        ):
            print(
                f"update={update} pns_signal_warning="
                "all_boundary_base_decisions_are_correct;"
                "evidence_removal_that_retains_the_base_decision_can_make_"
                "positive_pns_structurally_unobservable",
                flush=True,
            )
        for patient_index, sample in enumerate(update_samples, start=1):
            plan = plan_cache.get(sample.sample_id)
            if plan is None:
                plan = planner.plan(sample)
                plan_cache[sample.sample_id] = plan
                plan_generation_cache[sample.sample_id] = (
                    dict(planner.last_generation)
                    if planner.last_generation is not None
                    else None
                )
            reporter_calls_per_rollout = (
                1
                + args.pns_samples
                * (1 + len(plan.research_objectives))
            )
            research_rounds = len(plan.research_objectives)
            counterfactual_retrieval_calls = (
                0
                if args.pns_intervention == "removal"
                else research_rounds * (research_rounds + 1) // 2
            )
            planner_state = plan_generation_cache.get(sample.sample_id) or {}
            print(
                f"update={update} patient={patient_index}/"
                f"{len(update_samples)} sample={sample.sample_id} "
                f"planner_selection="
                f"{planner_state.get('selection', 'unknown')} "
                f"planner_primary_objectives="
                f"{planner_state.get('primary_objective_count', 0)} "
                f"planner_source_components="
                f"{planner_state.get('source_grounded_component_count', 0)} "
                f"research_rounds={research_rounds} "
                f"reporter_calls_per_rollout={reporter_calls_per_rollout} "
                f"counterfactual_retrieval_calls_per_rollout="
                f"{counterfactual_retrieval_calls}",
                flush=True,
            )
            rollouts: list[Rollout] = []
            for rollout_index in range(1, args.group_size + 1):
                started_at = time.monotonic()
                rollout = pipeline.rollout(sample, plan=plan)
                rollouts.append(rollout)
                completed_rollouts += 1
                rollout_pns = mean(
                    [step.pns_reward for step in rollout.steps]
                )
                rollout_pns_by_step = ",".join(
                    f"{step.evidence.step}:{step.pns_reward:.6f}"
                    for step in rollout.steps
                )
                monotonicity_violation_count = sum(
                    value < -1e-12
                    for step in rollout.steps
                    for value in (
                        step.signed_pns_difference or {}
                    ).values()
                )
                print(
                    f"update={update} "
                    f"rollout={completed_rollouts}/{total_rollouts} "
                    f"patient={patient_index}/{len(update_samples)} "
                    f"sample={sample.sample_id} "
                    f"group={rollout_index}/{args.group_size} "
                    f"reward={rollout.global_reward:.6f} "
                    f"boundary_bce={rollout.boundary_bce:.6f} "
                    f"ddi_rate={rollout.ddi_rate:.6f} "
                    f"pns_reward={rollout_pns:.6f} "
                    f"pns_by_step=[{rollout_pns_by_step}] "
                    f"pns_monotonicity_violations="
                    f"{monotonicity_violation_count} "
                    f"elapsed_seconds={time.monotonic() - started_at:.1f}",
                    flush=True,
                )
            attach_advantages(rollouts, args.group_size)
            recommendation = consensus_recommendation(
                sample,
                rollouts,
                ddi_matrix,
                update=update,
                patient_index=patient_index,
            )
            append_jsonl(recommendations_path, recommendation)
            print(
                f"update={update} patient={patient_index}/"
                f"{len(update_samples)} "
                f"sample={sample.sample_id} "
                "final_recommendation="
                + ",".join(
                    recommendation["recommended_medications"]
                )
                + " "
                f"consensus_ddi_rate="
                f"{recommendation['consensus_ddi_rate']:.6f}",
                flush=True,
            )
            groups.append(rollouts)
        update_planner_diagnostics = planner_distribution_diagnostics(
            [sample.sample_id for sample in update_samples],
            samples,
            plan_cache,
            plan_generation_cache,
        )
        cumulative_planner_diagnostics = planner_distribution_diagnostics(
            plan_cache.keys(),
            samples,
            plan_cache,
            plan_generation_cache,
        )
        if (
            cumulative_planner_diagnostics[
                "source_grounded_multi_component_rate"
            ]
            > 0.0
            and cumulative_planner_diagnostics[
                "final_multi_objective_rate"
            ]
            == 0.0
        ):
            print(
                "planner_design_warning=source_grounded_multi_component_"
                "cases_exist_but_no_final_plan_has_multiple_objectives",
                flush=True,
            )
        elif (
            cumulative_planner_diagnostics[
                "eligible_multi_boundary_sample_count"
            ]
            >= 10
            and cumulative_planner_diagnostics[
                "final_multi_objective_rate"
            ]
            == 0.0
        ):
            print(
                "planner_design_warning=at_least_10_multi_boundary_"
                "samples_were_audited_but_every_final_plan_has_only_O1",
                flush=True,
            )
        print(
            f"update={update} optimization_started "
            f"micro_batch={args.optimization_micro_batch} "
            f"epochs={args.optimization_epochs} "
            f"max_grad_norm={args.max_grad_norm} "
            f"target_kl={args.target_kl} "
            f"max_clipped_ratio_rate="
            f"{args.max_clipped_ratio_rate}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        statistics = optimize_questioner(
            questioner,
            groups,
            optimizer,
            args.optimization_micro_batch,
            args.optimization_epochs,
            args.max_grad_norm,
            args.target_kl,
            args.max_clipped_ratio_rate,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_rollouts = [
            rollout
            for group in groups
            for rollout in group
        ]
        all_steps = [
            step
            for rollout in all_rollouts
            for step in rollout.steps
        ]
        global_rewards = [
            rollout.global_reward for rollout in all_rollouts
        ]
        pns_rewards = [step.pns_reward for step in all_steps]
        signed_pns_differences = [
            value
            for step in all_steps
            for value in (step.signed_pns_difference or {}).values()
        ]
        full_pns_correctness = [
            value
            for step in all_steps
            for value in (step.full_correctness or {}).values()
        ]
        counterfactual_pns_correctness = [
            value
            for step in all_steps
            for value in (
                step.counterfactual_correctness or {}
            ).values()
        ]
        pns_semantic_repair_count = sum(
            rollout.full_pns_semantic_repair_count
            for rollout in all_rollouts
        ) + sum(
            step.counterfactual_pns_semantic_repair_count
            for step in all_steps
        )
        pns_repaired_sample_count = sum(
            rollout.full_pns_repaired_sample_count
            for rollout in all_rollouts
        ) + sum(
            step.counterfactual_pns_repaired_sample_count
            for step in all_steps
        )
        pns_no_evidence_retention_count = sum(
            rollout.full_pns_no_evidence_retention_count
            for rollout in all_rollouts
        ) + sum(
            step.counterfactual_pns_no_evidence_retention_count
            for step in all_steps
        )
        pns_nontrivial_repair_count = sum(
            rollout.full_pns_nontrivial_repair_count
            for rollout in all_rollouts
        ) + sum(
            step.counterfactual_pns_nontrivial_repair_count
            for step in all_steps
        )
        pns_nontrivially_repaired_sample_count = sum(
            rollout.full_pns_nontrivial_repaired_samples
            for rollout in all_rollouts
        ) + sum(
            step.counterfactual_pns_nontrivial_repaired_samples
            for step in all_steps
        )
        pns_reporter_sample_count = args.pns_samples * (
            len(all_rollouts) + len(all_steps)
        )
        advantages = [step.advantage for step in all_steps]
        normalized_questions = [
            " ".join(step.action.question.casefold().split())
            for step in all_steps
        ]
        question_jaccards = within_objective_question_jaccards(
            all_steps
        )
        metrics = {
            "update": update,
            "patient_count": len(update_samples),
            "group_size": args.group_size,
            "pns_samples": args.pns_samples,
            "pns_intervention": args.pns_intervention,
            "model_dtype": args.model_dtype,
            "trainable_parameter_dtypes": (
                trainable_parameter_dtypes
            ),
            "learning_rate": args.learning_rate,
            "target_kl": args.target_kl,
            "max_clipped_ratio_rate": (
                args.max_clipped_ratio_rate
            ),
            "reporter_max_conditional_shift": (
                args.reporter_max_conditional_shift
            ),
            "planner": update_planner_diagnostics,
            "planner_cumulative": cumulative_planner_diagnostics,
            **base_diagnostics,
            "research_action_count": sum(
                len(rollout.steps)
                for group in groups
                for rollout in group
            ),
            **statistics,
            "global_reward": mean(
                global_rewards
            ),
            "global_reward_std": float(np.std(global_rewards)),
            "boundary_bce": mean(
                [rollout.boundary_bce for rollout in all_rollouts]
            ),
            "ddi_rate": mean(
                [rollout.ddi_rate for rollout in all_rollouts]
            ),
            "ddi_positive_rollout_rate": mean(
                [
                    float(rollout.ddi_rate > 0.0)
                    for rollout in all_rollouts
                ]
            ),
            "pns_reward": mean(
                pns_rewards
            ),
            "pns_reward_std": float(np.std(pns_rewards)),
            "pns_reward_min": min(pns_rewards),
            "pns_reward_max": max(pns_rewards),
            "pns_positive_action_rate": mean(
                [float(value > 0.0) for value in pns_rewards]
            ),
            "pns_by_step": pns_by_step(all_rollouts),
            "pns_full_correctness_mean": mean(
                full_pns_correctness
            ),
            "pns_counterfactual_correctness_mean": mean(
                counterfactual_pns_correctness
            ),
            "pns_signed_difference_mean": mean(
                signed_pns_differences
            ),
            "pns_monotonicity_violation_rate": mean(
                [
                    float(value < -1e-12)
                    for value in signed_pns_differences
                ]
            ),
            "pns_reporter_sample_count": pns_reporter_sample_count,
            "pns_reporter_semantic_repair_count": (
                pns_semantic_repair_count
            ),
            "pns_reporter_semantic_repair_rate": (
                pns_repaired_sample_count
                / pns_reporter_sample_count
            ),
            "pns_reporter_no_evidence_retention_count": (
                pns_no_evidence_retention_count
            ),
            "pns_reporter_nontrivial_semantic_repair_count": (
                pns_nontrivial_repair_count
            ),
            "pns_reporter_nontrivial_semantic_repair_rate": (
                pns_nontrivially_repaired_sample_count
                / pns_reporter_sample_count
            ),
            "final_reporter_semantic_repair_count": sum(
                rollout.final_report_semantic_repair_count
                for rollout in all_rollouts
            ),
            "final_reporter_semantic_repair_rate": mean(
                [
                    float(
                        rollout.final_report_semantic_repair_count > 0
                    )
                    for rollout in all_rollouts
                ]
            ),
            "retriever_semantic_repair_count": sum(
                step.retriever_semantic_repair_count
                for step in all_steps
            ),
            "retriever_semantic_repair_rate": mean(
                [
                    float(step.retriever_semantic_repair_count > 0)
                    for step in all_steps
                ]
            ),
            "retriever_validation_rejection_count": sum(
                step.retriever_validation_rejection_count
                for step in all_steps
            ),
            "retriever_fallback_count": sum(
                int(step.retriever_fallback_used)
                for step in all_steps
            ),
            "retriever_fallback_rate": mean(
                [
                    float(step.retriever_fallback_used)
                    for step in all_steps
                ]
            ),
            "advantage_std": float(np.std(advantages)),
            "question_validation_rejection_count": sum(
                len(step.action.validation_rejections)
                for step in all_steps
            ),
            "question_resample_rate": mean(
                [
                    float(step.action.validation_attempts > 1)
                    for step in all_steps
                ]
            ),
            "question_semantic_repair_count": sum(
                len(step.action.semantic_repairs)
                for step in all_steps
            ),
            "question_semantic_repair_rate": mean(
                [
                    float(bool(step.action.semantic_repairs))
                    for step in all_steps
                ]
            ),
            "unique_question_rate": (
                len(set(normalized_questions))
                / len(normalized_questions)
            ),
            "mean_within_objective_question_jaccard": (
                mean_within_objective_question_jaccard(all_steps)
            ),
            "max_within_objective_question_jaccard": (
                max(question_jaccards) if question_jaccards else 0.0
            ),
            "near_duplicate_question_pair_rate": (
                mean(
                    [
                        float(value >= 0.8)
                        for value in question_jaccards
                    ]
                )
                if question_jaccards
                else 0.0
            ),
        }
        append_jsonl(metrics_path, metrics)
        print(
            f"update={update} loss={metrics['loss']:.6f} "
            f"reward={metrics['global_reward']:.6f} "
            f"reward_std={metrics['global_reward_std']:.6f} "
            f"pns={metrics['pns_reward']:.6f} "
            f"pns_positive_rate="
            f"{metrics['pns_positive_action_rate']:.6f} "
            f"pns_monotonicity_violation_rate="
            f"{metrics['pns_monotonicity_violation_rate']:.6f} "
            f"question_rejections="
            f"{metrics['question_validation_rejection_count']}",
            flush=True,
        )
        questioner_model.save_adapter(output_dir / "adapter-latest")
        if update % args.save_every == 0:
            questioner_model.save_adapter(
                output_dir / f"adapter-{update}"
            )

    questioner_model.save_adapter(output_dir / "adapter-final")


if __name__ == "__main__":
    main()
