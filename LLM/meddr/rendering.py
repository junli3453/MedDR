from __future__ import annotations

import json
from typing import Any

from meddr.schemas import EvidenceItem, PersonalizedPlan, ResearchSample


PRIVATE_FIELDS = {
    "account",
    "account_id",
    "admission_id",
    "address",
    "email",
    "encounter_id",
    "hadm",
    "hadm_id",
    "identifier",
    "label",
    "labels",
    "mrn",
    "patient_id",
    "patient_key",
    "patient_name",
    "phone",
    "record_id",
    "sample_id",
    "subject",
    "subject_id",
    "true_medications",
    "visit_id",
}


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitized(item)
            for key, item in value.items()
            if str(key).strip().casefold() not in PRIVATE_FIELDS
            and not str(key).strip().casefold().endswith("_identifier")
        }
    if isinstance(value, (list, tuple)):
        return [sanitized(item) for item in value]
    return value


def render(value: Any) -> str:
    return json.dumps(sanitized(value), ensure_ascii=True, indent=2)


def current_patient_context(sample: ResearchSample) -> str:
    return render(sample.current_visit)


def history_context(sample: ResearchSample) -> str:
    return render(sample.history)


def boundary_context(sample: ResearchSample) -> str:
    return render(
        {
            "probability_semantics": (
                "Each base_probability is the frozen deep model's probability "
                "that the medication belongs in the final recommended set."
            ),
            "medications": [
                {
                    "name": item.name,
                    "base_probability": round(item.probability, 6),
                }
                for item in sample.boundary_medications
            ],
        }
    )


def confident_context(sample: ResearchSample) -> str:
    return render(
        {
            "decision_semantics": (
                "These are frozen deep-model predictions, not observed current "
                "medication exposures."
            ),
            "deep_model_confidently_included": [
                item.name for item in sample.confident_accept
            ],
            "deep_model_confidently_excluded": [
                item.name for item in sample.confident_reject
            ],
        }
    )


def confidently_included_context(sample: ResearchSample) -> str:
    return render(
        {
            "decision_semantics": (
                "These are frozen deep-model predictions, not observed current "
                "medication exposures, and their probabilities remain unchanged."
            ),
            "deep_model_confidently_included": [
                item.name for item in sample.confident_accept
            ],
        }
    )


def similar_visit_context(sample: ResearchSample) -> str:
    return render(sample.similar_visits)


def ddi_context(sample: ResearchSample) -> str:
    return render(
        {
            "constraint_semantics": (
                "Each pair is a known adverse drug-drug interaction among "
                "candidate recommendations."
            ),
            "pairs": sample.ddi_constraints,
        }
    )


def candidate_contrast_data(sample: ResearchSample) -> dict[str, Any]:
    """Index source-grounded visit changes by boundary medication.

    This is a data representation aid for Planner structural review. It does
    not assign objectives or infer clinical relationships that are absent from
    the supplied longitudinal and cross-patient visits.
    """
    current_diagnoses = list(sample.current_visit.get("diagnoses", []))
    current_procedures = list(sample.current_visit.get("procedures", []))
    rows: list[dict[str, Any]] = []
    for medication in sample.boundary_medications:
        medication_key = medication.name.casefold()
        historical_sources = [
            visit
            for visit in sample.history
            if medication_key
            in {
                str(name).strip().casefold()
                for name in visit.get("medications", [])
            }
        ]
        cross_patient_contrasts: list[dict[str, Any]] = []
        supported_diagnosis_additions: list[str] = []
        supported_procedure_additions: list[str] = []
        seen_contrasts: set[tuple[str, str]] = set()
        for historical in historical_sources:
            historical_diagnoses = list(
                historical.get("diagnoses", [])
            )
            historical_procedures = list(
                historical.get("procedures", [])
            )
            historical_diagnosis_keys = {
                str(value).casefold() for value in historical_diagnoses
            }
            historical_procedure_keys = {
                str(value).casefold() for value in historical_procedures
            }
            for similar in sample.similar_visits:
                similar_diagnosis_keys = {
                    str(value).casefold()
                    for value in similar.get("diagnoses", [])
                }
                shared_anchor_diagnoses = [
                    value
                    for value in historical_diagnoses
                    if str(value).casefold() in similar_diagnosis_keys
                ]
                if not shared_anchor_diagnoses:
                    continue
                diagnosis_additions = [
                    value
                    for value in current_diagnoses
                    if str(value).casefold() in similar_diagnosis_keys
                    and str(value).casefold()
                    not in historical_diagnosis_keys
                ]
                similar_procedure_keys = {
                    str(value).casefold()
                    for value in similar.get("procedures", [])
                }
                procedure_additions = [
                    value
                    for value in current_procedures
                    if str(value).casefold() in similar_procedure_keys
                    and str(value).casefold()
                    not in historical_procedure_keys
                ]
                if not diagnosis_additions and not procedure_additions:
                    continue
                contrast_key = (
                    str(historical.get("source", "")),
                    str(similar.get("source", "")),
                )
                if contrast_key in seen_contrasts:
                    continue
                seen_contrasts.add(contrast_key)
                cross_patient_contrasts.append(
                    {
                        "same_patient_source": historical.get("source"),
                        "cross_patient_source": similar.get("source"),
                        "shared_anchor_diagnoses": (
                            shared_anchor_diagnoses
                        ),
                        "current_diagnosis_additions_also_present_in_"
                        "cross_patient_source": diagnosis_additions,
                        "current_procedure_additions_also_present_in_"
                        "cross_patient_source": procedure_additions,
                        "cross_patient_medication_pattern_semantics": (
                            "Contrastive context only; medication names are "
                            "intentionally omitted from this component."
                        ),
                    }
                )
                for value in diagnosis_additions:
                    if value not in supported_diagnosis_additions:
                        supported_diagnosis_additions.append(value)
                for value in procedure_additions:
                    if value not in supported_procedure_additions:
                        supported_procedure_additions.append(value)

        ddi_links = [
            list(pair)
            for pair in sample.ddi_constraints
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and medication_key
            in {str(pair[0]).casefold(), str(pair[1]).casefold()}
        ]
        rows.append(
            {
                "boundary_medication": medication.name,
                "same_patient_sources_recording_medication": (
                    historical_sources
                ),
                "source_grounded_cross_patient_contrasts": (
                    cross_patient_contrasts
                ),
                "candidate_linked_current_diagnosis_additions": (
                    supported_diagnosis_additions
                ),
                "candidate_linked_current_procedure_additions": (
                    supported_procedure_additions
                ),
                "known_ddi_links": ddi_links,
            }
        )
    return {
        "semantics": (
            "This inventory only reorganizes facts from the supplied visits. "
            "Distinct rows do not automatically require distinct objectives, "
            "and shared mechanisms or DDI links may be merged."
        ),
        "candidate_contrasts": rows,
    }


def plan_context(plan: PersonalizedPlan) -> str:
    return render(plan.to_dict())


def memory_context(memory: list[EvidenceItem]) -> str:
    return render([item.to_dict() for item in memory])
