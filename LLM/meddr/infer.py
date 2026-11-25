from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddr.agents import Planner, Questioner, Reporter
from meddr.constants import (
    DECISION_THRESHOLD,
    MAX_RESEARCH_ROUNDS,
    QUESTIONER_TEMPERATURE,
    QUESTIONER_TOP_P,
    REPORTER_MAX_CONDITIONAL_SHIFT,
)
from meddr.io import append_jsonl, load_pickle, load_samples
from meddr.modeling import FrozenChatModelView, LocalChatModel
from meddr.retrieval import Retriever
from meddr.rewards import ddi_rate, report_probabilities
from meddr.schemas import (
    EvidenceItem,
    PersonalizedPlan,
    ResearchSample,
    validate_and_align_ddi_constraints,
)


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description="Run MedDR inference with a trained Questioner adapter."
    )
    data_inputs = arguments.add_argument_group("Data inputs")
    data_inputs.add_argument(
        "--input-data",
        default="data/LLM-data/test.jsonl",
    )
    data_inputs.add_argument(
        "--ddi-matrix",
        default=os.getenv("MEDDR_DDI_MATRIX", "data/ready/ddi_A_final.pkl"),
        help="Preprocessed DDI adjacency file aligned with the medication vocabulary.",
    )
    arguments.add_argument(
        "--questioner-model",
        default="models/qwen3-8b",
    )
    arguments.add_argument(
        "--planner-model",
        default="models/qwen3-8b",
    )
    arguments.add_argument(
        "--reporter-model",
        default="models/qwen3-8b",
    )
    arguments.add_argument("--adapter-path", required=True)
    arguments.add_argument(
        "--retriever-model",
        default=os.getenv("MEDDR_RETRIEVER_MODEL"),
    )
    arguments.add_argument(
        "--retriever-base-url",
        default=os.getenv("MEDDR_RETRIEVER_ENDPOINT"),
    )
    arguments.add_argument("--retriever-timeout", type=float, default=120.0)
    arguments.add_argument("--retriever-max-retries", type=int, default=3)
    arguments.add_argument(
        "--retriever-thinking",
        choices=("enabled", "disabled"),
    )
    arguments.add_argument(
        "--output-file",
        default="outputs/meddr-inference/predictions.jsonl",
    )
    arguments.add_argument("--rollouts", type=int, default=1)
    arguments.add_argument("--limit", type=int, default=0)
    arguments.add_argument(
        "--device-map",
        default=os.getenv("MEDDR_DEVICE_MAP", "balanced"),
    )
    arguments.add_argument(
        "--model-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default=os.getenv("MEDDR_MODEL_DTYPE", "float16"),
    )
    arguments.add_argument("--questioner-max-tokens", type=int, default=256)
    arguments.add_argument("--planner-max-tokens", type=int, default=768)
    arguments.add_argument("--reporter-max-tokens", type=int, default=1024)
    arguments.add_argument("--retriever-max-tokens", type=int, default=1024)
    arguments.add_argument(
        "--reporter-max-conditional-shift",
        type=float,
        default=REPORTER_MAX_CONDITIONAL_SHIFT,
    )
    arguments.add_argument("--seed", type=int, default=42)
    return arguments


def run_research_rollout(
    sample: ResearchSample,
    plan: PersonalizedPlan,
    questioner: Questioner,
    retriever: Retriever,
    reporter: Reporter,
) -> dict[str, Any]:
    rounds = len(plan.research_objectives)
    if not 1 <= rounds <= MAX_RESEARCH_ROUNDS:
        raise ValueError(
            "The Planner objective count is outside the supported range."
        )
    memory: list[EvidenceItem] = []
    for step in range(1, rounds + 1):
        action = questioner.ask(
            sample,
            plan,
            memory,
            step,
            score_logprob=False,
        )
        evidence = EvidenceItem(
            step=step,
            objective_id=action.objective_id,
            question=action.question,
            answer=retriever.retrieve(action.question),
        )
        memory.append(evidence)
    report = reporter.report(
        sample,
        plan,
        memory,
        temperature=0.0,
    )
    return {
        "trajectory": [item.to_dict() for item in memory],
        "report": report,
    }


def consensus_prediction(
    sample: ResearchSample,
    plan: PersonalizedPlan,
    rollouts: list[dict[str, Any]],
    ddi_matrix: Any,
) -> dict[str, Any]:
    reports = [row["report"] for row in rollouts]
    probabilities = [report_probabilities(report) for report in reports]
    boundary_rows: list[dict[str, Any]] = []
    selected_ids = set(sample.confident_accept_ids)
    for medication in sample.boundary_medications:
        values = [row[medication.name] for row in probabilities]
        mean_probability = float(np.mean(values))
        recommended = mean_probability >= DECISION_THRESHOLD
        if recommended:
            selected_ids.add(medication.identifier)
        boundary_rows.append(
            {
                "name": medication.name,
                "base_probability": medication.probability,
                "mean_adjusted_probability": mean_probability,
                "recommended": recommended,
            }
        )
    ordered_candidates = sample.medication_candidates
    recommended_names = [
        medication.name
        for medication in ordered_candidates
        if medication.identifier in selected_ids
    ]
    return {
        "sample_id": sample.sample_id,
        "plan": plan.to_dict(),
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
        "boundary_medications": boundary_rows,
        "confidently_included_medications": [
            medication.name for medication in sample.confident_accept
        ],
        "recommended_medications": recommended_names,
        "consensus_ddi_rate": ddi_rate(selected_ids, ddi_matrix),
        "selection_threshold": DECISION_THRESHOLD,
    }


def main() -> None:
    args = parser().parse_args()
    if args.rollouts < 1:
        raise ValueError("--rollouts must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    output_path = Path(args.output_file)
    if output_path.exists():
        raise FileExistsError(
            "The output file already exists. Choose a new --output-file."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_samples(args.input_data)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No inference samples were loaded.")
    ddi_matrix = load_pickle(args.ddi_matrix)
    validate_and_align_ddi_constraints(samples, ddi_matrix)

    questioner_model = LocalChatModel(
        args.questioner_model,
        device_map=args.device_map,
        trainable=False,
        adapter_path=args.adapter_path,
        torch_dtype=args.model_dtype,
    )
    base_view = FrozenChatModelView(questioner_model)
    planner_model = (
        base_view
        if args.planner_model == args.questioner_model
        else LocalChatModel(
            args.planner_model,
            device_map=args.device_map,
            trainable=False,
            torch_dtype=args.model_dtype,
        )
    )
    reporter_model = (
        base_view
        if args.reporter_model == args.questioner_model
        else planner_model
        if args.reporter_model == args.planner_model
        else LocalChatModel(
            args.reporter_model,
            device_map=args.device_map,
            trainable=False,
            torch_dtype=args.model_dtype,
        )
    )
    planner = Planner(planner_model, max_new_tokens=args.planner_max_tokens)
    questioner = Questioner(
        questioner_model,
        temperature=QUESTIONER_TEMPERATURE,
        top_p=QUESTIONER_TOP_P,
        max_new_tokens=args.questioner_max_tokens,
    )
    reporter = Reporter(
        reporter_model,
        max_new_tokens=args.reporter_max_tokens,
        max_conditional_shift=args.reporter_max_conditional_shift,
    )
    retriever = Retriever(
        args.retriever_model,
        base_url=args.retriever_base_url,
        max_tokens=args.retriever_max_tokens,
        timeout=args.retriever_timeout,
        max_retries=args.retriever_max_retries,
        thinking=args.retriever_thinking,
    )

    for index, sample in enumerate(samples, start=1):
        plan = planner.plan(sample)
        rollouts = [
            run_research_rollout(
                sample,
                plan,
                questioner,
                retriever,
                reporter,
            )
            for _ in range(args.rollouts)
        ]
        prediction = consensus_prediction(
            sample,
            plan,
            rollouts,
            ddi_matrix,
        )
        append_jsonl(output_path, prediction)
        print(
            f"sample={index}/{len(samples)} "
            f"sample_id={sample.sample_id} "
            "recommended_medications="
            + ",".join(prediction["recommended_medications"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
