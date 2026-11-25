from __future__ import annotations

from meddr.schemas import (
    PersonalizedPlan,
    ResearchObjective,
    ResearchSample,
)


PNS_INTERVENTIONS = ("removal", "rm", "tox")


def joined(values: list[str] | tuple[str, ...]) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def removal_of_personalization_plan(
    objective: ResearchObjective,
) -> PersonalizedPlan:
    medications = joined(objective.affected_candidates)
    return PersonalizedPlan(
        plan_summary=(
            "Use a general patient-related medication research objective. "
            "Do not use the Personalized Plan's longitudinal or cross-patient "
            "contrast."
        ),
        research_objectives=(
            ResearchObjective(
                objective_id=objective.objective_id,
                patient_contrast=(
                    "Personalized longitudinal and cross-patient differences "
                    "are intentionally unavailable for this intervention."
                ),
                affected_candidates=objective.affected_candidates,
                research_objective=(
                    "Clarify general clinical and pharmacological evidence "
                    f"relevant to {medications} for the stated current patient "
                    "context."
                ),
            ),
        ),
    )


def toxic_objective_plan(
    source: ResearchSample,
    objective_id: str,
) -> PersonalizedPlan:
    diagnoses = source.current_visit.get("diagnoses", [])
    medications = [
        medication.name for medication in source.boundary_medications
    ]
    diagnosis_text = joined(diagnoses) if diagnoses else "the stated condition"
    medication_text = joined(medications)
    return PersonalizedPlan(
        plan_summary=(
            "Use the research objective derived from a different patient for "
            "this intervention."
        ),
        research_objectives=(
            ResearchObjective(
                objective_id=objective_id,
                patient_contrast=(
                    "The unrelated source patient has a current context of "
                    f"{diagnosis_text}."
                ),
                affected_candidates=tuple(medications),
                research_objective=(
                    "Clarify general indication and safety evidence for "
                    f"{medication_text} in the unrelated source context."
                ),
            ),
        ),
    )


def select_toxic_source(
    target: ResearchSample,
    candidates: tuple[ResearchSample, ...],
) -> ResearchSample:
    alternatives = [
        candidate
        for candidate in candidates
        if candidate.sample_id != target.sample_id
    ]
    if alternatives:
        target_medications = {
            medication.name.casefold()
            for medication in target.boundary_medications
        }
        target_diagnoses = {
            diagnosis.casefold()
            for diagnosis in target.current_visit.get("diagnoses", [])
        }
        return min(
            alternatives,
            key=lambda candidate: (
                len(
                    target_medications
                    & {
                        medication.name.casefold()
                        for medication in candidate.boundary_medications
                    }
                ),
                len(
                    target_diagnoses
                    & {
                        diagnosis.casefold()
                        for diagnosis in candidate.current_visit.get(
                            "diagnoses",
                            [],
                        )
                    }
                ),
                candidate.sample_id,
            ),
        )
    raise ValueError(
        "The TOX intervention requires at least two distinct samples."
    )
