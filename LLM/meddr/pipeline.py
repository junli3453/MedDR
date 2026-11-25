from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meddr.agents import Planner, Questioner, Reporter
from meddr.constants import (
    DDI_PENALTY_WEIGHT,
    MAX_RESEARCH_ROUNDS,
    PNS_INTERVENTION,
    PNS_REPORTER_SAMPLES,
)
from meddr.interventions import (
    PNS_INTERVENTIONS,
    removal_of_personalization_plan,
    select_toxic_source,
    toxic_objective_plan,
)
from meddr.modeling import QuestionAction
from meddr.retrieval import Retriever
from meddr.rewards import (
    boundary_bce,
    ddi_rate,
    pns_diagnostics,
    predicted_medications,
)
from meddr.schemas import EvidenceItem, PersonalizedPlan, ResearchSample


@dataclass
class RolloutStep:
    action: QuestionAction
    evidence: EvidenceItem
    pns_reward: float = 0.0
    advantage: float = 0.0
    full_correctness: dict[str, float] | None = None
    counterfactual_correctness: dict[str, float] | None = None
    signed_pns_difference: dict[str, float] | None = None
    positive_pns_contribution: dict[str, float] | None = None
    counterfactual_pns_semantic_repair_count: int = 0
    counterfactual_pns_repaired_sample_count: int = 0
    counterfactual_pns_no_evidence_retention_count: int = 0
    counterfactual_pns_nontrivial_repair_count: int = 0
    counterfactual_pns_nontrivial_repaired_samples: int = 0
    retriever_semantic_repair_count: int = 0
    retriever_validation_rejection_count: int = 0
    retriever_fallback_used: bool = False


@dataclass
class Rollout:
    plan: PersonalizedPlan
    steps: list[RolloutStep]
    final_report: dict[str, Any]
    global_reward: float
    boundary_bce: float
    ddi_rate: float
    full_pns_semantic_repair_count: int = 0
    full_pns_repaired_sample_count: int = 0
    full_pns_no_evidence_retention_count: int = 0
    full_pns_nontrivial_repair_count: int = 0
    full_pns_nontrivial_repaired_samples: int = 0
    final_report_semantic_repair_count: int = 0


class MedDRPipeline:
    def __init__(
        self,
        planner: Planner,
        questioner: Questioner,
        retriever: Retriever,
        reporter: Reporter,
        *,
        ddi_matrix: Any,
        rounds: int = MAX_RESEARCH_ROUNDS,
        pns_samples: int = PNS_REPORTER_SAMPLES,
        pns_intervention: str = PNS_INTERVENTION,
        intervention_samples: tuple[ResearchSample, ...] = (),
        ddi_weight: float = DDI_PENALTY_WEIGHT,
    ) -> None:
        if not 1 <= rounds <= MAX_RESEARCH_ROUNDS:
            raise ValueError(
                f"Research rounds must be between 1 and {MAX_RESEARCH_ROUNDS}."
            )
        if pns_samples < 1:
            raise ValueError("PNS Reporter samples must be positive.")
        if ddi_matrix is None:
            raise ValueError("A DDI adjacency matrix is required.")
        if pns_intervention not in PNS_INTERVENTIONS:
            raise ValueError(
                "PNS intervention must be one of: "
                + ", ".join(PNS_INTERVENTIONS)
                + "."
            )
        if pns_intervention == "tox" and len(intervention_samples) < 2:
            raise ValueError(
                "The TOX intervention requires at least two samples."
            )
        self.planner = planner
        self.questioner = questioner
        self.retriever = retriever
        self.reporter = reporter
        self.ddi_matrix = ddi_matrix
        self.rounds = rounds
        self.pns_samples = pns_samples
        self.pns_intervention = pns_intervention
        self.intervention_samples = intervention_samples
        self.ddi_weight = ddi_weight

    def counterfactual_trajectory(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan,
        original_steps: list[RolloutStep],
        intervention_index: int,
    ) -> list[EvidenceItem]:
        if not 0 <= intervention_index < len(original_steps):
            raise ValueError("Counterfactual intervention index is invalid.")

        if self.pns_intervention == "removal":
            memory: list[EvidenceItem] = []
            for index, original in enumerate(original_steps):
                evidence = EvidenceItem(
                    step=original.evidence.step,
                    objective_id=original.evidence.objective_id,
                    question=original.evidence.question,
                    answer=(
                        None
                        if index == intervention_index
                        else original.evidence.answer
                    ),
                )
                memory.append(evidence)
            return memory

        memory = [
            original.evidence
            for original in original_steps[:intervention_index]
        ]
        original_intervention = original_steps[intervention_index]
        objective_id = original_intervention.action.objective_id
        if self.pns_intervention == "rm":
            source_sample = sample
            original_objective = next(
                objective
                for objective in plan.research_objectives
                if objective.objective_id == objective_id
            )
            intervention_plan = removal_of_personalization_plan(
                original_objective
            )
        else:
            source_sample = select_toxic_source(
                sample,
                self.intervention_samples,
            )
            intervention_plan = toxic_objective_plan(
                source_sample,
                objective_id,
            )

        action = self.questioner.ask(
            source_sample,
            intervention_plan,
            memory,
            original_intervention.evidence.step,
            score_logprob=False,
        )
        answer = self.retriever.retrieve(action.question)
        evidence = EvidenceItem(
            step=original_intervention.evidence.step,
            objective_id=objective_id,
            question=action.question,
            answer=answer,
        )
        memory.append(evidence)

        for step_number in range(
            original_intervention.evidence.step + 1,
            len(original_steps) + 1,
        ):
            action = self.questioner.ask(
                sample,
                plan,
                memory,
                step_number,
                score_logprob=False,
            )
            answer = self.retriever.retrieve(action.question)
            evidence = EvidenceItem(
                step=step_number,
                objective_id=action.objective_id,
                question=action.question,
                answer=answer,
            )
            memory.append(evidence)
        return memory

    def pns_reports(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan,
        memory: list[EvidenceItem],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        reports: list[dict[str, Any]] = []
        semantic_repair_count = 0
        semantic_repaired_sample_count = 0
        no_evidence_retention_count = 0
        nontrivial_repair_count = 0
        nontrivially_repaired_sample_count = 0
        for _ in range(self.pns_samples):
            reports.append(self.reporter.sample_for_pns(sample, plan, memory))
            generation = dict(self.reporter.last_generation or {})
            repairs = generation.get("semantic_repairs", [])
            semantic_repair_count += len(repairs)
            semantic_repaired_sample_count += int(bool(repairs))
            no_evidence_repairs = [
                repair
                for repair in repairs
                if "because no direct medication-linked retrieval" in repair
            ]
            nontrivial_repairs = [
                repair
                for repair in repairs
                if repair not in no_evidence_repairs
            ]
            no_evidence_retention_count += len(no_evidence_repairs)
            nontrivial_repair_count += len(nontrivial_repairs)
            nontrivially_repaired_sample_count += int(
                bool(nontrivial_repairs)
            )
        return reports, {
            "semantic_repair_count": semantic_repair_count,
            "semantic_repaired_sample_count": (
                semantic_repaired_sample_count
            ),
            "no_evidence_retention_count": no_evidence_retention_count,
            "nontrivial_repair_count": nontrivial_repair_count,
            "nontrivially_repaired_sample_count": (
                nontrivially_repaired_sample_count
            ),
        }

    def rollout(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan | None = None,
    ) -> Rollout:
        sample.require_training_labels()
        personalized_plan = plan or self.planner.plan(sample)
        rollout_rounds = len(personalized_plan.research_objectives)
        if not 1 <= rollout_rounds <= self.rounds:
            raise ValueError(
                "The Planner objective count must be between 1 and the "
                f"configured maximum of {self.rounds}."
            )
        memory: list[EvidenceItem] = []
        steps: list[RolloutStep] = []
        for step_number in range(1, rollout_rounds + 1):
            action = self.questioner.ask(
                sample,
                personalized_plan,
                memory,
                step_number,
            )
            answer = self.retriever.retrieve(action.question)
            evidence = EvidenceItem(
                step=step_number,
                objective_id=action.objective_id,
                question=action.question,
                answer=answer,
            )
            memory.append(evidence)
            steps.append(
                RolloutStep(
                    action=action,
                    evidence=evidence,
                    retriever_semantic_repair_count=len(
                        (
                            self.retriever.last_generation or {}
                        ).get("semantic_repairs", [])
                    ),
                    retriever_validation_rejection_count=len(
                        (
                            self.retriever.last_generation or {}
                        ).get("validation_rejections", [])
                    ),
                    retriever_fallback_used=bool(
                        (
                            self.retriever.last_generation or {}
                        ).get("fallback_used", False)
                    ),
                )
            )

        final_report = self.reporter.report(
            sample,
            personalized_plan,
            memory,
            temperature=0.0,
        )
        final_generation_state = dict(
            self.reporter.last_generation or {}
        )
        bce = boundary_bce(sample, final_report)
        safety = ddi_rate(
            predicted_medications(sample, final_report),
            self.ddi_matrix,
        )
        global_reward = -bce - self.ddi_weight * safety

        full_reports, full_report_generations = self.pns_reports(
            sample,
            personalized_plan,
            memory,
        )
        for intervention_index, rollout_step in enumerate(steps):
            counterfactual_memory = self.counterfactual_trajectory(
                sample,
                personalized_plan,
                steps,
                intervention_index,
            )
            (
                counterfactual_reports,
                counterfactual_report_generations,
            ) = self.pns_reports(
                sample,
                personalized_plan,
                counterfactual_memory,
            )
            if counterfactual_report_generations is not None:
                counterfactual_stats = counterfactual_report_generations
                rollout_step.counterfactual_pns_semantic_repair_count = int(
                    counterfactual_stats.get(
                        "semantic_repair_count",
                        0,
                    )
                )
                rollout_step.counterfactual_pns_repaired_sample_count = int(
                    counterfactual_stats.get(
                        "semantic_repaired_sample_count",
                        0,
                    )
                )
                rollout_step.counterfactual_pns_no_evidence_retention_count = (
                    int(
                        counterfactual_stats.get(
                            "no_evidence_retention_count",
                            0,
                        )
                    )
                )
                rollout_step.counterfactual_pns_nontrivial_repair_count = int(
                    counterfactual_stats.get(
                        "nontrivial_repair_count",
                        0,
                    )
                )
                rollout_step.counterfactual_pns_nontrivial_repaired_samples = (
                    int(
                        counterfactual_stats.get(
                            "nontrivially_repaired_sample_count",
                            0,
                        )
                    )
                )
            diagnostics = pns_diagnostics(
                sample,
                full_reports,
                counterfactual_reports,
            )
            rollout_step.full_correctness = diagnostics[
                "full_correctness"
            ]
            rollout_step.counterfactual_correctness = diagnostics[
                "counterfactual_correctness"
            ]
            rollout_step.signed_pns_difference = diagnostics[
                "signed_difference"
            ]
            rollout_step.positive_pns_contribution = diagnostics[
                "positive_contribution"
            ]
            contributions = diagnostics["positive_contribution"]
            rollout_step.pns_reward = (
                sum(contributions.values()) / len(contributions)
            )

        return Rollout(
            plan=personalized_plan,
            steps=steps,
            final_report=final_report,
            global_reward=global_reward,
            boundary_bce=bce,
            ddi_rate=safety,
            full_pns_semantic_repair_count=(
                int(
                    full_report_generations.get(
                        "semantic_repair_count",
                        0,
                    )
                )
                if full_report_generations is not None
                else 0
            ),
            full_pns_repaired_sample_count=(
                int(
                    full_report_generations.get(
                        "semantic_repaired_sample_count",
                        0,
                    )
                )
                if full_report_generations is not None
                else 0
            ),
            full_pns_no_evidence_retention_count=(
                int(
                    full_report_generations.get(
                        "no_evidence_retention_count",
                        0,
                    )
                )
                if full_report_generations is not None
                else 0
            ),
            full_pns_nontrivial_repair_count=(
                int(
                    full_report_generations.get(
                        "nontrivial_repair_count",
                        0,
                    )
                )
                if full_report_generations is not None
                else 0
            ),
            full_pns_nontrivial_repaired_samples=(
                int(
                    full_report_generations.get(
                        "nontrivially_repaired_sample_count",
                        0,
                    )
                )
                if full_report_generations is not None
                else 0
            ),
            final_report_semantic_repair_count=len(
                final_generation_state.get(
                    "semantic_repairs",
                    [],
                )
            ),
        )
