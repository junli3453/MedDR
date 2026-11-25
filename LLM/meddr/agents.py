from __future__ import annotations

import json
import re
from typing import Any

from meddr.constants import (
    MAX_RESEARCH_ROUNDS,
    PNS_REPORTER_TEMPERATURE,
    PNS_REPORTER_VALIDATION_ATTEMPTS,
    QUESTIONER_TEMPERATURE,
    QUESTIONER_TOP_P,
    REPORTER_MAX_CONDITIONAL_SHIFT,
    SEMANTIC_VALIDATION_ATTEMPTS,
)
from meddr.modeling import LocalChatModel, QuestionAction
from meddr.prompts import (
    PLANNER_COMPONENT_SYSTEM,
    PLANNER_SYSTEM,
    QUESTIONER_SYSTEM,
    REPORTER_SYSTEM,
    planner_component_prompt,
    planner_prompt,
    questioner_prompt,
    reporter_prompt,
)
from meddr.rendering import candidate_contrast_data
from meddr.schemas import (
    EvidenceItem,
    PersonalizedPlan,
    ResearchObjective,
    ResearchSample,
)
from meddr.semantic_validation import (
    generic_evidence_objective,
    normalized_text,
    repair_question_exposure_language,
    repair_planner_ddi_objective,
    repair_planner_ddi_exposure_language,
    repair_planner_exposure_language,
    repair_planner_foreign_medication_context,
    repair_planner_monitoring_objective,
    repair_planner_research_objective,
    repair_reporter_context_as_proof,
    validate_no_inferred_current_exposure,
    validate_planner_objective,
    validate_question,
)


def require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} must contain exactly: {', '.join(sorted(expected))}."
        )


def _grounding_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2
        and token
        not in {
            "and",
            "for",
            "from",
            "patient",
            "patients",
            "the",
            "this",
            "visit",
            "with",
        }
    }


def _recover_affected_candidates(
    raw_text: str,
    sample: ResearchSample,
) -> tuple[str, ...]:
    """Ground a malformed affected-candidate field in supplied visit data."""
    raw_normalized = normalized_text(raw_text).casefold()
    raw_tokens = _grounding_tokens(raw_text)
    scores: dict[str, int] = {}
    for medication in sample.boundary_medications:
        score = 0
        for visit in (*sample.history, *sample.similar_visits):
            medications = {
                str(name).strip().casefold()
                for name in visit.get("medications", [])
            }
            if medication.name.casefold() not in medications:
                continue
            for field in ("diagnoses", "procedures"):
                for value in visit.get(field, []):
                    term = normalized_text(str(value)).casefold()
                    term_tokens = _grounding_tokens(term)
                    if term and term in raw_normalized:
                        score += 4 + len(term_tokens)
                    else:
                        score += len(term_tokens & raw_tokens)
        scores[medication.name] = score
    maximum = max(scores.values(), default=0)
    if maximum <= 0:
        return ()
    return tuple(
        name for name, score in scores.items() if score == maximum
    )


class Planner:
    def __init__(
        self,
        model: LocalChatModel,
        max_new_tokens: int = 768,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.last_generation: dict[str, Any] | None = None

    def plan(self, sample: ResearchSample) -> PersonalizedPlan:
        user_prompt = planner_prompt(sample)
        primary_prompt = self.model.format_prompt(
            PLANNER_SYSTEM,
            user_prompt,
        )
        primary_completion: str | None = None
        primary_plan: PersonalizedPlan | None = None
        primary_repairs: list[str] = []
        validation_rejections: list[str] = []
        try:
            primary_prompt, primary_completion, parsed = self.model.generate(
                PLANNER_SYSTEM,
                user_prompt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
            )
            primary_plan, primary_repairs = self._parse_plan(parsed, sample)
        except ValueError as exc:
            validation_rejections.append(
                f"Primary Planner draft: {exc}"
            )

        components = self._source_grounded_components(sample)
        expected_partition = {
            frozenset(component["affected_candidates"])
            for component in components
        }
        primary_partition = (
            {
                frozenset(objective.affected_candidates)
                for objective in primary_plan.research_objectives
            }
            if primary_plan is not None
            else set()
        )
        partition_mismatch = bool(components) and (
            primary_partition != expected_partition
        )
        reconstruction_requested = (
            primary_plan is None
            or partition_mismatch
            or any("DDI" in repair for repair in primary_repairs)
        )
        component_generations: list[dict[str, Any]] = []
        component_objectives: list[ResearchObjective] = []
        reconstruction_failed = False

        if reconstruction_requested and components:
            for component_index, component in enumerate(components, start=1):
                affected_candidates = tuple(
                    component["affected_candidates"]
                )
                component_user_prompt = planner_component_prompt(
                    user_prompt,
                    component,
                    affected_candidates,
                )
                component_prompt = self.model.format_prompt(
                    PLANNER_COMPONENT_SYSTEM,
                    component_user_prompt,
                )
                component_completion: str | None = None
                component_repairs: list[str] = []
                component_error: str | None = None
                component_plan: PersonalizedPlan | None = None
                try:
                    (
                        component_prompt,
                        component_completion,
                        parsed,
                    ) = self.model.generate(
                        PLANNER_COMPONENT_SYSTEM,
                        component_user_prompt,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=self.max_new_tokens,
                    )
                    component_plan, component_repairs = self._parse_plan(
                        parsed,
                        sample,
                    )
                    if len(component_plan.research_objectives) != 1:
                        raise ValueError(
                            "A source-grounded component must produce exactly "
                            "one research objective."
                        )
                    generated = component_plan.research_objectives[0]
                    combined_text = (
                        f"{generated.patient_contrast} "
                        f"{generated.research_objective}"
                    ).casefold()
                    missing_names = [
                        name
                        for name in affected_candidates
                        if name.casefold() not in combined_text
                    ]
                    research_objective = generated.research_objective
                    if missing_names:
                        research_objective += (
                            " Clarify the shared component-level evidence "
                            "implications for "
                            + ", ".join(affected_candidates)
                            + "."
                        )
                        component_repairs.append(
                            "Restored explicit references to every candidate "
                            "in the source-grounded component."
                        )
                    patient_contrast = self._component_patient_contrast(
                        component
                    )
                    if patient_contrast != generated.patient_contrast:
                        component_repairs.append(
                            "Rebuilt the patient contrast from the supplied "
                            "diagnoses, procedures, visit sources, and DDI "
                            "links without assuming medication exposure."
                        )
                    objective = ResearchObjective(
                        objective_id=f"O{component_index}",
                        patient_contrast=patient_contrast,
                        affected_candidates=affected_candidates,
                        research_objective=research_objective,
                    )
                    validate_planner_objective(objective, sample)
                    component_objectives.append(objective)
                except ValueError as exc:
                    component_error = str(exc)
                    validation_rejections.append(
                        "Source-grounded Planner component "
                        f"{component_index}: {exc}"
                    )
                    reconstruction_failed = True
                component_generations.append(
                    {
                        "component_index": component_index,
                        "affected_candidates": list(affected_candidates),
                        "formatted_prompt": component_prompt,
                        "raw_completion": component_completion,
                        "semantic_repairs": component_repairs,
                        "validation_error": component_error,
                    }
                )
                if reconstruction_failed:
                    break

        reconstruction_selected = bool(
            reconstruction_requested
            and components
            and not reconstruction_failed
            and len(component_objectives) == len(components)
        )
        if reconstruction_selected:
            selected_plan = PersonalizedPlan(
                plan_summary=self._source_grounded_plan_summary(components),
                research_objectives=tuple(component_objectives),
            )
            selected_prompt = component_generations[-1]["formatted_prompt"]
            selected_completion = json.dumps(
                selected_plan.to_dict(),
                ensure_ascii=False,
            )
            selected_repairs = [
                repair
                for generation in component_generations
                for repair in generation["semantic_repairs"]
            ]
            selection = "source_grounded_planner"
        elif primary_plan is not None:
            selected_plan = primary_plan
            selected_prompt = primary_prompt
            selected_completion = primary_completion
            selected_repairs = primary_repairs
            selection = "primary"
        else:
            selected_plan = self._minimum_plan(sample)
            selected_prompt = primary_prompt
            selected_completion = primary_completion
            selected_repairs = [
                "Used one deterministic minimum Personalized Plan after "
                "Planner generation failed."
            ]
            selection = "minimum_fallback"

        self.last_generation = {
            "formatted_prompt": selected_prompt,
            "raw_completion": selected_completion,
            "semantic_repairs": selected_repairs,
            "validation_rejections": validation_rejections,
            "fallback_used": selection == "minimum_fallback",
            "selection": selection,
            "primary_objective_count": (
                len(primary_plan.research_objectives)
                if primary_plan is not None
                else 0
            ),
            "final_objective_count": len(
                selected_plan.research_objectives
            ),
            "source_grounded_component_count": len(components),
            "source_grounded_component_candidates": [
                list(component["affected_candidates"])
                for component in components
            ],
            "source_grounded_partition_mismatch": partition_mismatch,
            "source_grounded_reconstruction_requested": (
                reconstruction_requested
            ),
            "source_grounded_reconstruction_selected": (
                reconstruction_selected
            ),
            "source_grounded_reconstruction_failed": reconstruction_failed,
            "source_grounded_component_generations": component_generations,
            "structural_review_requested": reconstruction_requested,
            "structural_review_selected": reconstruction_selected,
            "structural_review_signals": (
                ["Primary objective partition differs from the "
                 "source-grounded patient-difference components."]
                if partition_mismatch
                else []
            ),
            "structural_review_attempt_count": len(component_generations),
            "structural_review_unresolved": bool(
                reconstruction_requested and not reconstruction_selected
            ),
            "primary_generation": {
                "formatted_prompt": primary_prompt,
                "raw_completion": primary_completion,
                "semantic_repairs": primary_repairs,
            },
            "structural_review_generation": (
                component_generations[-1]
                if component_generations
                else None
            ),
            "structural_review_generations": component_generations,
        }
        return selected_plan

    @staticmethod
    def _minimum_plan(sample: ResearchSample) -> PersonalizedPlan:
        candidates = tuple(
            medication.name
            for medication in sample.boundary_medications
        )
        return PersonalizedPlan(
            plan_summary=(
                "Use patient-specific clinical and pharmacological evidence "
                "to evaluate the frozen boundary medication predictions."
            ),
            research_objectives=(
                ResearchObjective(
                    objective_id="O1",
                    patient_contrast=(
                        "The current clinical context is contrasted with the "
                        "supplied historical and cross-patient visit context "
                        "for the frozen boundary predictions."
                    ),
                    affected_candidates=candidates,
                    research_objective=generic_evidence_objective(candidates),
                ),
            ),
        )

    @staticmethod
    def _list_phrase(values: list[str] | tuple[str, ...]) -> str:
        items = [str(value) for value in values if str(value)]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    @staticmethod
    def _lower_initial(value: str) -> str:
        return value[:1].lower() + value[1:] if value else value

    @staticmethod
    def _component_patient_contrast(
        component: dict[str, Any],
    ) -> str:
        candidates = tuple(component["affected_candidates"])
        diagnoses: list[str] = []
        procedures: list[str] = []
        historical_sources: list[str] = []
        cross_patient_sources: list[str] = []
        for row in component["candidate_contrasts"]:
            for value in row.get(
                "candidate_linked_current_diagnosis_additions", []
            ):
                if value not in diagnoses:
                    diagnoses.append(str(value))
            for value in row.get(
                "candidate_linked_current_procedure_additions", []
            ):
                if value not in procedures:
                    procedures.append(str(value))
            for visit in row.get(
                "same_patient_sources_recording_medication", []
            ):
                source = str(visit.get("source", "")).strip()
                if source and source not in historical_sources:
                    historical_sources.append(source)
            for contrast in row.get(
                "source_grounded_cross_patient_contrasts", []
            ):
                source = str(
                    contrast.get("cross_patient_source", "")
                ).strip()
                if source and source not in cross_patient_sources:
                    cross_patient_sources.append(source)

        candidate_text = Planner._list_phrase(candidates)
        differences = [
            Planner._lower_initial(value)
            for value in diagnoses + procedures
        ]
        sentences: list[str] = []
        if differences:
            source_text = Planner._list_phrase(historical_sources)
            source_clause = (
                "Compared with the same-patient historical "
                + ("source " if len(historical_sources) == 1 else "sources ")
                + f"{source_text}, "
                if source_text
                else "Compared with the supplied same-patient history, "
            )
            sentences.append(
                source_clause
                + "the current visit adds "
                + Planner._list_phrase(differences)
                + f", a clinical-state difference relevant to the boundary "
                f"candidate{'s' if len(candidates) != 1 else ''} "
                + candidate_text
                + "."
            )
        else:
            sentences.append(
                "The supplied current and longitudinal clinical contexts "
                "define an unresolved patient-specific contrast relevant to "
                f"the boundary candidate{'s' if len(candidates) != 1 else ''} "
                + candidate_text
                + "."
            )
        if cross_patient_sources:
            sentences.append(
                "These clinical differences are also represented in "
                + Planner._list_phrase(cross_patient_sources)
                + (", which is retained " if len(cross_patient_sources) == 1
                   else ", which are retained ")
                + "as contrastive context rather than "
                "target-patient clinical proof."
            )
        ddi_links = component.get("known_ddi_links", [])
        if ddi_links:
            pairs = [
                Planner._list_phrase(tuple(str(name) for name in pair))
                for pair in ddi_links
            ]
            sentences.append(
                "The supplied DDI constraint links "
                + Planner._list_phrase(pairs)
                + " as conditional boundary candidates without establishing "
                "observed co-exposure."
            )
        return " ".join(sentences)

    @staticmethod
    def _source_grounded_plan_summary(
        components: list[dict[str, Any]],
    ) -> str:
        candidates: list[str] = []
        differences: list[str] = []
        for component in components:
            for name in component["affected_candidates"]:
                if name not in candidates:
                    candidates.append(str(name))
            for row in component["candidate_contrasts"]:
                for field in (
                    "candidate_linked_current_diagnosis_additions",
                    "candidate_linked_current_procedure_additions",
                ):
                    for value in row.get(field, []):
                        normalized_value = Planner._lower_initial(str(value))
                        if normalized_value not in differences:
                            differences.append(normalized_value)
        difference_text = (
            "The current clinical-state differences in "
            + Planner._list_phrase(differences)
            if differences
            else "The supplied patient-specific differences"
        )
        return (
            difference_text
            + " relative to same-patient history and cross-patient "
            "comparators define the main uncertainties for the boundary "
            "candidates "
            + Planner._list_phrase(candidates)
            + "."
        )

    @staticmethod
    def _source_grounded_components(
        sample: ResearchSample,
    ) -> list[dict[str, Any]]:
        inventory = candidate_contrast_data(sample)
        rows = list(inventory["candidate_contrasts"])
        order = {
            medication.name: index
            for index, medication in enumerate(sample.boundary_medications)
        }
        row_lookup = {
            str(row["boundary_medication"]): row for row in rows
        }
        features: dict[str, set[str]] = {}
        for name, row in row_lookup.items():
            features[name] = {
                "diagnosis:" + str(value).casefold()
                for value in row.get(
                    "candidate_linked_current_diagnosis_additions",
                    [],
                )
            } | {
                "procedure:" + str(value).casefold()
                for value in row.get(
                    "candidate_linked_current_procedure_additions",
                    [],
                )
            }
        ddi_links = [
            tuple(str(value) for value in pair)
            for pair in sample.ddi_constraints
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        ddi_names = {name for pair in ddi_links for name in pair}
        active = [
            medication.name
            for medication in sample.boundary_medications
            if features.get(medication.name) or medication.name in ddi_names
        ]
        if not active:
            return []

        parent = {name: name for name in active}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(first: str, second: str) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for index, first in enumerate(active):
            for second in active[index + 1 :]:
                if features[first] & features[second]:
                    union(first, second)
        for first, second in ddi_links:
            if first in parent and second in parent:
                union(first, second)

        grouped: dict[str, list[str]] = {}
        for name in active:
            grouped.setdefault(find(name), []).append(name)
        components: list[dict[str, Any]] = []
        for names in grouped.values():
            names.sort(key=order.__getitem__)
            name_set = set(names)
            component_ddi = [
                list(pair)
                for pair in ddi_links
                if set(pair) <= name_set
            ]
            components.append(
                {
                    "affected_candidates": tuple(names),
                    "candidate_contrasts": [
                        row_lookup[name] for name in names
                    ],
                    "shared_source_grounded_features": sorted(
                        set.intersection(
                            *(features[name] for name in names)
                        )
                        if len(names) > 1
                        else features[names[0]]
                    ),
                    "known_ddi_links": component_ddi,
                }
            )
        components.sort(
            key=lambda component: min(
                order[name]
                for name in component["affected_candidates"]
            )
        )
        if len(components) <= MAX_RESEARCH_ROUNDS:
            return components
        ranked = sorted(
            components,
            key=lambda component: (
                -int(bool(component["known_ddi_links"])),
                -sum(
                    len(row.get(
                        "candidate_linked_current_diagnosis_additions", []
                    ))
                    + len(row.get(
                        "candidate_linked_current_procedure_additions", []
                    ))
                    for row in component["candidate_contrasts"]
                ),
                min(
                    order[name]
                    for name in component["affected_candidates"]
                ),
            ),
        )[:MAX_RESEARCH_ROUNDS]
        return sorted(
            ranked,
            key=lambda component: min(
                order[name]
                for name in component["affected_candidates"]
            ),
        )

    def _parse_plan(
        self,
        parsed: dict[str, Any],
        sample: ResearchSample,
    ) -> tuple[PersonalizedPlan, list[str]]:
        require_exact_keys(
            parsed,
            {"plan_summary", "research_objectives"},
            "Planner output",
        )
        plan_summary, semantic_repairs = repair_planner_exposure_language(
            str(parsed["plan_summary"]).strip(),
            sample,
        )
        plan_summary, summary_medication_repairs = (
            repair_planner_foreign_medication_context(
                plan_summary,
                sample,
                tuple(
                    medication.name
                    for medication in sample.boundary_medications
                ),
            )
        )
        semantic_repairs.extend(summary_medication_repairs)
        raw_objectives = parsed["research_objectives"]
        if (
            not plan_summary
            or not isinstance(raw_objectives, list)
            or not raw_objectives
        ):
            raise ValueError(
                "The Planner must return a non-empty plan and objectives."
            )
        validate_no_inferred_current_exposure(
            plan_summary,
            sample,
            context="Planner plan_summary",
        )
        objectives: list[ResearchObjective] = []
        seen: set[str] = set()
        for raw in raw_objectives:
            if not isinstance(raw, dict):
                raise ValueError("Every research objective must be an object.")
            require_exact_keys(
                raw,
                {
                    "objective_id",
                    "patient_contrast",
                    "affected_candidates",
                    "research_objective",
                },
                "Planner research objective",
            )
            objective_id = str(raw["objective_id"]).strip()
            contrast, contrast_repairs = repair_planner_exposure_language(
                str(raw["patient_contrast"]).strip(),
                sample,
            )
            objective = str(raw["research_objective"]).strip()
            affected = raw["affected_candidates"]
            if (
                not objective_id
                or not contrast
                or not objective
                or not isinstance(affected, list)
                or not affected
            ):
                raise ValueError("Planner objectives must be non-empty.")
            if objective_id in seen:
                original_id = objective_id
                suffix = len(objectives) + 1
                objective_id = f"O{suffix}"
                while objective_id in seen:
                    suffix += 1
                    objective_id = f"O{suffix}"
                semantic_repairs.append(
                    f"Renamed duplicate Planner objective {original_id} to "
                    f"{objective_id}."
                )
            boundary_names = tuple(
                item.name for item in sample.boundary_medications
            )
            raw_candidates = tuple(
                str(name).strip()
                for name in affected
                if str(name).strip()
            )
            matched_candidates: list[str] = []
            for raw_name in raw_candidates:
                raw_key = re.sub(r"[^a-z0-9]+", "", raw_name.casefold())
                for boundary_name in boundary_names:
                    boundary_key = re.sub(
                        r"[^a-z0-9]+",
                        "",
                        boundary_name.casefold(),
                    )
                    if boundary_key and boundary_key in raw_key:
                        matched_candidates.append(boundary_name)
            candidates = tuple(dict.fromkeys(matched_candidates))
            removed_candidates = [
                name
                for name in raw_candidates
                if not any(
                    re.sub(
                        r"[^a-z0-9]+",
                        "",
                        boundary_name.casefold(),
                    )
                    in re.sub(r"[^a-z0-9]+", "", name.casefold())
                    for boundary_name in boundary_names
                )
            ]
            if removed_candidates:
                semantic_repairs.append(
                    "Removed non-boundary affected candidates from "
                    f"{objective_id}: {', '.join(removed_candidates)}."
                )
            if not candidates:
                combined_raw = f"{contrast} {objective}".casefold()
                candidates = tuple(
                    name
                    for name in boundary_names
                    if name.casefold() in combined_raw
                )
            if not candidates:
                candidates = _recover_affected_candidates(
                    " ".join((*raw_candidates, contrast, objective)),
                    sample,
                )
                if candidates:
                    semantic_repairs.append(
                        "Recovered affected boundary candidates for "
                        f"{objective_id} from medication-linked diagnoses "
                        "and procedures in the supplied visits."
                    )
            if not candidates and len(boundary_names) == 1:
                candidates = boundary_names
                semantic_repairs.append(
                    f"Restored the sole boundary medication as the affected "
                    f"candidate for {objective_id}."
                )
            if not candidates:
                semantic_repairs.append(
                    f"Dropped {objective_id} because it did not identify an "
                    "affected boundary medication."
                )
                continue
            contrast, contrast_medication_repairs = (
                repair_planner_foreign_medication_context(
                    contrast,
                    sample,
                    candidates,
                )
            )
            objective, objective_repairs = (
                repair_planner_exposure_language(
                    objective,
                    sample,
                )
            )
            objective, objective_medication_repairs = (
                repair_planner_foreign_medication_context(
                    objective,
                    sample,
                    candidates,
                )
            )
            objective, decision_repairs = (
                repair_planner_research_objective(
                    objective,
                    candidates,
                )
            )
            objective, ddi_repairs = repair_planner_ddi_objective(
                objective,
                sample,
                candidates,
            )
            objective, ddi_exposure_repairs = (
                repair_planner_ddi_exposure_language(
                    objective,
                    sample,
                    candidates,
                )
            )
            objective, monitoring_repairs = (
                repair_planner_monitoring_objective(
                    objective,
                    candidates,
                )
            )
            combined_casefold = f"{contrast} {objective}".casefold()
            relevant_candidates = tuple(
                name
                for name in candidates
                if name.casefold() in combined_casefold
            )
            if not relevant_candidates:
                objective = generic_evidence_objective(candidates)
                relevant_candidates = candidates
                semantic_repairs.append(
                    f"Reframed {objective_id} to explicitly address its "
                    "affected boundary medication."
                )
            if relevant_candidates != candidates:
                removed = [
                    name for name in candidates
                    if name not in relevant_candidates
                ]
                semantic_repairs.append(
                    f"Removed unrelated affected candidates from "
                    f"{objective_id}: {', '.join(removed)}."
                )
            semantic_repairs.extend(contrast_repairs)
            semantic_repairs.extend(contrast_medication_repairs)
            semantic_repairs.extend(objective_repairs)
            semantic_repairs.extend(objective_medication_repairs)
            semantic_repairs.extend(decision_repairs)
            semantic_repairs.extend(ddi_repairs)
            semantic_repairs.extend(ddi_exposure_repairs)
            semantic_repairs.extend(monitoring_repairs)
            research_objective = ResearchObjective(
                objective_id=objective_id,
                patient_contrast=contrast,
                affected_candidates=relevant_candidates,
                research_objective=objective,
            )
            validate_planner_objective(research_objective, sample)
            objectives.append(research_objective)
            seen.add(objective_id)
        if len(objectives) > MAX_RESEARCH_ROUNDS:
            raise ValueError(
                "The Planner returned more research objectives than the "
                f"maximum of {MAX_RESEARCH_ROUNDS}."
            )
        if not objectives:
            raise ValueError(
                "Planner semantic validation removed every research objective."
            )
        return PersonalizedPlan(
            plan_summary,
            tuple(objectives),
        ), semantic_repairs


class Questioner:
    def __init__(
        self,
        model: LocalChatModel,
        *,
        temperature: float = QUESTIONER_TEMPERATURE,
        top_p: float = QUESTIONER_TOP_P,
        max_new_tokens: int = 256,
        validation_attempts: int = SEMANTIC_VALIDATION_ATTEMPTS,
    ) -> None:
        if validation_attempts < 1:
            raise ValueError("Questioner validation_attempts must be positive.")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.validation_attempts = validation_attempts

    def ask(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan,
        memory: list[EvidenceItem],
        step: int,
        *,
        score_logprob: bool = True,
    ) -> QuestionAction:
        valid_objectives = {
            item.objective_id for item in plan.research_objectives
        }
        resolved_objectives = {item.objective_id for item in memory}
        unresolved_objectives = valid_objectives - resolved_objectives
        if not unresolved_objectives:
            raise ValueError("The Personalized Plan has no unresolved objective.")
        previous_questions = {
            normalized_text(item.question).casefold() for item in memory
        }
        user_prompt = questioner_prompt(sample, plan, memory, step)
        rejections: list[str] = []
        for attempt in range(1, self.validation_attempts + 1):
            try:
                prompt, completion, parsed = self.model.generate(
                    QUESTIONER_SYSTEM,
                    user_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                )
                action = self._parse_action(
                    parsed,
                    prompt=prompt,
                    completion=completion,
                    unresolved_objectives=unresolved_objectives,
                    previous_questions=previous_questions,
                    plan=plan,
                    sample=sample,
                    validation_attempts=attempt,
                    validation_rejections=tuple(rejections),
                    score_logprob=score_logprob,
                )
            except ValueError as exc:
                rejections.append(str(exc))
                continue
            return action
        objective_id = sorted(unresolved_objectives)[0]
        objective = next(
            item
            for item in plan.research_objectives
            if item.objective_id == objective_id
        )
        diagnoses = ", ".join(sample.current_visit["diagnoses"])
        patient_context = diagnoses or "the stated clinical conditions"
        candidates = " and ".join(objective.affected_candidates)
        question = (
            f"For a patient with {patient_context}, what patient-specific "
            "clinical and pharmacological evidence is needed to evaluate "
            f"the predicted conditional recommendation of {candidates}?"
        )
        prompt = self.model.format_prompt(
            QUESTIONER_SYSTEM,
            user_prompt,
        )
        completion = json.dumps(
            {
                "objective_id": objective_id,
                "question": question,
            },
            ensure_ascii=False,
        )
        old_logprob = (
            float(
                self.model.score_logprobs(
                    [prompt],
                    [completion],
                ).cpu()[0]
            )
            if score_logprob
            else 0.0
        )
        return QuestionAction(
            prompt=prompt,
            completion=completion,
            objective_id=objective_id,
            question=question,
            old_logprob=old_logprob,
            validation_attempts=self.validation_attempts,
            validation_rejections=tuple(rejections),
            semantic_repairs=(
                "Used a deterministic conditional Questioner action after "
                "generation or semantic validation failed.",
            ),
        )

    def _parse_action(
        self,
        parsed: dict[str, Any],
        *,
        prompt: str,
        completion: str,
        unresolved_objectives: set[str],
        previous_questions: set[str],
        plan: PersonalizedPlan,
        sample: ResearchSample,
        validation_attempts: int,
        validation_rejections: tuple[str, ...],
        score_logprob: bool,
    ) -> QuestionAction:
        require_exact_keys(
            parsed,
            {"objective_id", "question"},
            "Questioner output",
        )
        objective_id = str(parsed["objective_id"]).strip()
        question, semantic_repairs = repair_question_exposure_language(
            str(parsed["question"]).strip(),
            sample,
        )
        normalized_question = normalized_text(question).casefold()
        if objective_id not in unresolved_objectives:
            raise ValueError(
                "Questioner selected an objective that is already resolved "
                "or absent from the Personalized Plan."
            )
        if not question:
            raise ValueError("Questioner returned an empty question.")
        if normalized_question in previous_questions:
            raise ValueError("Questioner repeated an accepted question.")
        objective = next(
            item
            for item in plan.research_objectives
            if item.objective_id == objective_id
        )
        validate_question(question, objective, sample)
        old_logprob = (
            float(
                self.model.score_logprobs(
                    [prompt],
                    [completion],
                ).cpu()[0]
            )
            if score_logprob
            else 0.0
        )
        return QuestionAction(
            prompt=prompt,
            completion=completion,
            objective_id=objective_id,
            question=question,
            old_logprob=old_logprob,
            validation_attempts=validation_attempts,
            validation_rejections=validation_rejections,
            semantic_repairs=tuple(semantic_repairs),
        )


class Reporter:
    def __init__(
        self,
        model: LocalChatModel,
        *,
        max_new_tokens: int = 1024,
        max_conditional_shift: float = (
            REPORTER_MAX_CONDITIONAL_SHIFT
        ),
        pns_validation_attempts: int = (
            PNS_REPORTER_VALIDATION_ATTEMPTS
        ),
    ) -> None:
        if not 0.0 < max_conditional_shift <= 1.0:
            raise ValueError(
                "Reporter max_conditional_shift must be in (0, 1]."
            )
        if pns_validation_attempts != 1:
            raise ValueError(
                "Reporter PNS sampling must use exactly one accepted "
                "generation per Eq. (15) rollout."
            )
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_conditional_shift = max_conditional_shift
        self.pns_validation_attempts = pns_validation_attempts
        self.last_generation: dict[str, Any] | None = None

    def report(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan,
        memory: list[EvidenceItem],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        user_prompt = reporter_prompt(sample, plan, memory)
        try:
            prompt, completion, parsed = self.model.generate(
                REPORTER_SYSTEM,
                user_prompt,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
            )
            report = validate_report(parsed, sample)
            report, semantic_repairs = repair_reporter_context_as_proof(
                report,
                sample,
                plan,
                memory,
                max_conditional_shift=self.max_conditional_shift,
            )
            validation_rejections: list[str] = []
            fallback_used = False
        except ValueError as exc:
            prompt = self.model.format_prompt(
                REPORTER_SYSTEM,
                user_prompt,
            )
            completion = ""
            report = {
                "clinical_summary": (
                    "No valid generated report was available; frozen "
                    "boundary probabilities were retained."
                ),
                "medications": [
                    {
                        "name": medication.name,
                        "base_probability": medication.probability,
                        "adjusted_probability": medication.probability,
                        "evidence": (
                            "No valid objective-linked evidence report was "
                            "available."
                        ),
                        "reason": (
                            "The frozen base probability is retained because "
                            "the generated report could not be validated."
                        ),
                    }
                    for medication in sample.boundary_medications
                ],
                "safety_alerts": [],
            }
            semantic_repairs = [
                "Retained frozen boundary probabilities after Reporter "
                "generation or validation failed."
            ]
            validation_rejections = [str(exc)]
            fallback_used = True
        self.last_generation = {
            "formatted_prompt": prompt,
            "raw_completion": completion,
            "semantic_repairs": semantic_repairs,
            "validation_rejections": validation_rejections,
            "fallback_used": fallback_used,
        }
        return report

    def sample_for_pns(
        self,
        sample: ResearchSample,
        plan: PersonalizedPlan,
        memory: list[EvidenceItem],
    ) -> dict[str, Any]:
        report = self.report(
            sample,
            plan,
            memory,
            temperature=PNS_REPORTER_TEMPERATURE,
        )
        generation = dict(self.last_generation or {})
        generation["validation_attempts"] = 1
        generation.setdefault("validation_rejections", [])
        generation["validation_exhausted"] = False
        self.last_generation = generation
        return report


def validate_report(
    parsed: dict[str, Any],
    sample: ResearchSample,
) -> dict[str, Any]:
    require_exact_keys(
        parsed,
        {"clinical_summary", "medications", "safety_alerts"},
        "Reporter output",
    )
    if not isinstance(parsed["medications"], list):
        raise ValueError("Reporter medications must be a list.")
    expected = {item.name: item for item in sample.boundary_medications}
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in parsed["medications"]:
        if not isinstance(row, dict):
            raise ValueError("Each Reporter medication must be an object.")
        require_exact_keys(
            row,
            {
                "name",
                "base_probability",
                "adjusted_probability",
                "evidence",
                "reason",
            },
            "Reporter medication",
        )
        name = str(row["name"]).strip()
        if name not in expected or name in seen:
            raise ValueError(
                "Reporter must include every boundary medication exactly once."
            )
        base = float(row["base_probability"])
        adjusted = float(row["adjusted_probability"])
        if abs(base - expected[name].probability) > 1e-6:
            raise ValueError("Reporter base_probability must match the frozen model.")
        if not 0.0 <= adjusted <= 1.0:
            raise ValueError("Reporter adjusted_probability must be in [0, 1].")
        evidence = str(row["evidence"]).strip()
        reason = str(row["reason"]).strip()
        if not evidence or not reason:
            raise ValueError("Reporter evidence and reason must be non-empty.")
        normalized.append(
            {
                "name": name,
                "base_probability": base,
                "adjusted_probability": adjusted,
                "evidence": evidence,
                "reason": reason,
            }
        )
        seen.add(name)
    if seen != set(expected):
        raise ValueError(
            "Reporter must include every boundary medication exactly once."
        )
    alerts = parsed["safety_alerts"]
    if not isinstance(alerts, list):
        raise ValueError("Reporter safety_alerts must be a list.")
    clinical_summary = str(parsed["clinical_summary"]).strip()
    if not clinical_summary:
        raise ValueError("Reporter clinical_summary must be non-empty.")
    return {
        "clinical_summary": clinical_summary,
        "medications": normalized,
        "safety_alerts": [str(item).strip() for item in alerts],
    }
