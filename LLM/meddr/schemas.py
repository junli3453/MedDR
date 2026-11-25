from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meddr.constants import BOUNDARY_MAX, BOUNDARY_MIN, SIMILAR_PATIENTS


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text_list(value: Any) -> list[str]:
    return [str(item).strip() for item in as_list(value) if str(item).strip()]


def diagnosis_set(visit: dict[str, Any]) -> set[str]:
    return {item.casefold() for item in text_list(visit.get("diagnoses"))}


def diagnosis_jaccard(current: dict[str, Any], candidate: dict[str, Any]) -> float:
    current_values = diagnosis_set(current)
    candidate_values = diagnosis_set(candidate)
    union = current_values | candidate_values
    return len(current_values & candidate_values) / len(union) if union else 0.0


def select_cross_patient_visits(
    current: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_patient_key: str,
    limit: int = SIMILAR_PATIENTS,
) -> list[dict[str, Any]]:
    """Keep the highest-Jaccard visit per other patient, then the global top K."""
    best_by_patient: dict[str, tuple[float, int, dict[str, Any]]] = {}
    for position, raw_visit in enumerate(candidates):
        visit = dict(raw_visit)
        patient_key = str(visit.get("patient_key", "")).strip()
        if not patient_key:
            raise ValueError(
                "Every similar-visit candidate requires patient_key."
            )
        if target_patient_key and patient_key == target_patient_key:
            continue
        score = diagnosis_jaccard(current, visit)
        if score <= 0.0:
            continue
        previous = best_by_patient.get(patient_key)
        row = (score, -position, visit)
        if previous is None or row[:2] > previous[:2]:
            best_by_patient[patient_key] = row
    ranked = sorted(
        best_by_patient.values(),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )[:limit]
    selected: list[dict[str, Any]] = []
    for rank, (score, _, visit) in enumerate(ranked, start=1):
        selected.append(
            {
                "source": f"cross_patient_visit_{rank}",
                "similarity": score,
                "diagnoses": text_list(visit.get("diagnoses")),
                "procedures": text_list(visit.get("procedures")),
                "medications": text_list(visit.get("medications")),
            }
        )
    return selected


@dataclass(frozen=True)
class MedicationCandidate:
    identifier: int
    name: str
    probability: float
    label: int | None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MedicationCandidate":
        probability = float(row["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Medication probabilities must be in [0, 1].")
        label = row.get("label")
        if label is not None and int(label) not in (0, 1):
            raise ValueError("Medication labels must be binary.")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("Every medication requires a readable name.")
        return cls(
            identifier=int(row["id"]),
            name=name,
            probability=probability,
            label=None if label is None else int(label),
        )


@dataclass
class ResearchSample:
    sample_id: str
    current_visit: dict[str, list[str]]
    history: list[dict[str, Any]]
    boundary_medications: list[MedicationCandidate]
    confident_accept: list[MedicationCandidate]
    confident_reject: list[MedicationCandidate]
    similar_visits: list[dict[str, Any]] = field(default_factory=list)
    ddi_constraints: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ResearchSample":
        patient = row.get("patient")
        base_outputs = row.get("base_model_outputs")
        if not isinstance(patient, dict) or not isinstance(base_outputs, dict):
            raise ValueError("Each sample requires patient and base_model_outputs objects.")
        current_raw = patient.get("current_visit")
        if not isinstance(current_raw, dict):
            raise ValueError("patient.current_visit must be an object.")
        current = {
            "diagnoses": text_list(current_raw.get("diagnoses")),
            "procedures": text_list(current_raw.get("procedures")),
        }
        history: list[dict[str, Any]] = []
        for index, item in enumerate(as_list(patient.get("history")), start=1):
            if not isinstance(item, dict):
                continue
            history.append(
                {
                    "source": f"historical_visit_{index}",
                    "diagnoses": text_list(item.get("diagnoses")),
                    "procedures": text_list(item.get("procedures")),
                    "medications": text_list(item.get("medications")),
                }
            )
        medications = [
            MedicationCandidate.from_dict(item)
            for item in as_list(base_outputs.get("medications"))
            if isinstance(item, dict)
        ]
        if len({item.identifier for item in medications}) != len(medications):
            raise ValueError("Medication identifiers must be unique within a sample.")
        if len({item.name for item in medications}) != len(medications):
            raise ValueError("Medication names must be unique within a sample.")
        boundary = [
            item
            for item in medications
            if BOUNDARY_MIN <= item.probability <= BOUNDARY_MAX
        ]
        if not boundary:
            raise ValueError(
                "Each sample requires at least one medication in the closed "
                f"[{BOUNDARY_MIN}, {BOUNDARY_MAX}] boundary interval."
            )
        target_patient_key = str(patient.get("patient_key", "")).strip()
        if not target_patient_key:
            raise ValueError("patient.patient_key is required.")
        cross_candidates = [
            dict(item)
            for item in as_list(row.get("similar_visit_candidates"))
            if isinstance(item, dict)
        ]
        return cls(
            sample_id=str(row.get("sample_id", "")).strip() or "sample",
            current_visit=current,
            history=history,
            boundary_medications=boundary,
            confident_accept=[
                item for item in medications if item.probability > BOUNDARY_MAX
            ],
            confident_reject=[
                item for item in medications if item.probability < BOUNDARY_MIN
            ],
            similar_visits=select_cross_patient_visits(
                current,
                cross_candidates,
                target_patient_key,
            ),
            ddi_constraints=as_list(row.get("ddi_constraints")),
        )

    def require_training_labels(self) -> None:
        missing = [item.name for item in self.boundary_medications if item.label is None]
        if missing:
            raise ValueError(
                "Boundary medications are missing training labels: "
                + ", ".join(missing)
            )

    @property
    def boundary_names(self) -> set[str]:
        return {item.name for item in self.boundary_medications}

    @property
    def confident_accept_ids(self) -> set[int]:
        return {item.identifier for item in self.confident_accept}

    @property
    def medication_candidates(self) -> list[MedicationCandidate]:
        return (
            self.boundary_medications
            + self.confident_accept
            + self.confident_reject
        )


def validate_and_align_ddi_constraints(
    samples: list[ResearchSample],
    ddi_matrix: Any,
) -> None:
    shape = getattr(ddi_matrix, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("The DDI matrix must be a square two-dimensional matrix.")
    vocabulary_size = int(shape[0])
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for sample in samples:
        for medication in sample.medication_candidates:
            identifier = medication.identifier
            name = medication.name
            if not 0 <= identifier < vocabulary_size:
                raise ValueError(
                    f"Medication id {identifier} is outside the DDI matrix."
                )
            previous_name = id_to_name.get(identifier)
            if previous_name is not None and previous_name != name:
                raise ValueError(
                    f"Medication id {identifier} maps to both "
                    f"{previous_name!r} and {name!r}."
                )
            previous_id = name_to_id.get(name)
            if previous_id is not None and previous_id != identifier:
                raise ValueError(
                    f"Medication {name!r} maps to both ids "
                    f"{previous_id} and {identifier}."
                )
            id_to_name[identifier] = name
            name_to_id[name] = identifier

    for sample in samples:
        potentially_selected = (
            sample.confident_accept + sample.boundary_medications
        )
        expected: list[list[str]] = []
        expected_keys: set[frozenset[str]] = set()
        for index, first in enumerate(potentially_selected):
            for second in potentially_selected[index + 1 :]:
                interacts = bool(
                    ddi_matrix[first.identifier][second.identifier]
                ) or bool(ddi_matrix[second.identifier][first.identifier])
                if not interacts:
                    continue
                expected.append([first.name, second.name])
                expected_keys.add(frozenset((first.name, second.name)))

        supplied_keys: set[frozenset[str]] = set()
        for pair in sample.ddi_constraints:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    "Every DDI constraint must contain exactly two names."
                )
            first_name = str(pair[0]).strip()
            second_name = str(pair[1]).strip()
            if (
                first_name not in name_to_id
                or second_name not in name_to_id
                or first_name == second_name
            ):
                raise ValueError(
                    f"Invalid DDI constraint: {first_name!r}, {second_name!r}."
                )
            supplied_keys.add(frozenset((first_name, second_name)))

        if supplied_keys and supplied_keys != expected_keys:
            raise ValueError(
                f"DDI constraints for sample {sample.sample_id!r} do not "
                "match the DDI matrix and medication vocabulary."
            )
        sample.ddi_constraints = expected


@dataclass(frozen=True)
class ResearchObjective:
    objective_id: str
    patient_contrast: str
    affected_candidates: tuple[str, ...]
    research_objective: str


@dataclass(frozen=True)
class PersonalizedPlan:
    plan_summary: str
    research_objectives: tuple[ResearchObjective, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_summary": self.plan_summary,
            "research_objectives": [
                {
                    "objective_id": item.objective_id,
                    "patient_contrast": item.patient_contrast,
                    "affected_candidates": list(item.affected_candidates),
                    "research_objective": item.research_objective,
                }
                for item in self.research_objectives
            ],
        }


@dataclass(frozen=True)
class EvidenceItem:
    step: int
    objective_id: str
    question: str
    answer: str | None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "objective_id": self.objective_id,
            "question": self.question,
        }
        if self.answer is not None:
            row["answer"] = self.answer
        return row
