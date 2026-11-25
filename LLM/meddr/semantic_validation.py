from __future__ import annotations

import re
from typing import Any

from meddr.constants import REPORTER_MAX_CONDITIONAL_SHIFT
from meddr.schemas import (
    EvidenceItem,
    PersonalizedPlan,
    ResearchObjective,
    ResearchSample,
)


PRESCRIBING_DECISION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:inclusion|exclusion)\b",
        r"\bre-?introduc(?:e|es|ed|ing|tion)\b",
        r"\b(?:initiat(?:e|es|ed|ing|ion)|discontinu(?:e|es|ed|ing|ation))\b",
        r"\b(?:should|would)\b.{0,80}\b(?:be\s+)?"
        r"(?:prescrib(?:e|ed)|recommend(?:ed)?|include(?:d)?|exclude(?:d)?)\b",
        r"\bsupport(?:s|ed|ing)?\b.{0,60}\b"
        r"(?:prescribing|prescription|recommending|recommendation)\b",
        r"\b(?:benefit|benefits)\b.{0,80}\boutweigh\b.{0,80}\brisk",
        r"\bshould\b.{0,80}\b(?:receive|use|take|start|stop|avoid|continue|switch)\b",
    )
)

CURRENT_EXPOSURE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\babsence of\b.{0,40}\b(?:drug|medication|therapy|treatment)\b"
        r".{0,80}\b(?:current state|current visit|current medication(?:s| "
        r"list| regimen)?)\b",
        r"\bcurrent (?:state|visit|medication(?:s| list| regimen)?)\b"
        r".{0,100}\b(?:absent|absence|missing|does not include|do not include|"
        r"not include|without)\b.{0,40}\b"
        r"(?:drug|medication|therapy|treatment)\b",
        r"\bcurrent patient\b.{0,120}\b(?:currently taking|currently receiving|"
        r"currently using|concurrent use|coadministration|co-administ(?:er|ration))\b",
        r"\bcurrent patient\b.{0,100}\bhas\s+no\b.{0,80}\bin medications\b",
        r"\b(?:currently taking|currently receiving|currently using)\b",
        r"\b(?:the\s+)?current patient['\u2019]s\s+"
        r"(?:use|receipt|administration)\s+of\b",
        r"\b(?:the\s+)?current (?:treatment|medication) plan\b"
        r".{0,100}\b(?:includes?|involves?|contains?)\b",
        r"\b(?:is|are|was|were)\s+(?:currently\s+)?prescribed\s+"
        r"(?:in|for|to)\s+(?:the\s+)?current patient\b",
        r"\bcurrent absence of\b.{0,80}\b"
        r"(?:anticoagulant|drug|medication|therapy|treatment)\b",
        r"\bcurrent medication list\b",
    )
)

DDI_PATTERN = re.compile(
    r"\b(?:DDI|drug[-\s]drug interaction)\b",
    re.IGNORECASE,
)

REPORTER_MONITORING_PROOF_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:presence|availability|use|performance)\s+of\b.{0,80}"
        r"\bmonitor(?:ing|ed)?\b.{0,160}\b"
        r"(?:provides?|supplies?|offers?|yields?)\b.{0,80}\b"
        r"(?:data|evidence|support|proof)\b",
        r"\bmonitoring\s+(?:context|procedure|presence)\b.{0,160}\b"
        r"(?:supports?|justifies?|increases?|decreases?)\b.{0,100}\b"
        r"(?:probability|inclusion|exclusion|use)\b",
        r"\bprobability\b.{0,100}\b(?:increas(?:e|ed)|decreas(?:e|ed))\b"
        r".{0,160}\bmonitor(?:ing|ed)?\b",
    )
)

CROSS_PATIENT_SOURCE = (
    r"(?:cross[-_\s]*patient(?:[-_\s]*visit)?(?:[-_\s]*\d+)?"
    r"|similar[-_\s]*visit(?:[-_\s]*\d+)?)"
)

REPORTER_CROSS_PATIENT_PROOF_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b{CROSS_PATIENT_SOURCE}\b.{{0,180}}\b"
        r"(?:contradict(?:s|ed|ing)?|corroborat(?:e|es|ed|ing)|"
        r"demonstrat(?:e|es|ed|ing)|indicat(?:e|es|ed|ing)|"
        r"prov(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|support(?:s|ed|ing)?)\b"
        r".{0,120}\b"
        r"(?:us(?:e|ed|es|ing)|inclusion|exclusion|probability|"
        r"indicat(?:e|ed|es|ing|ion)|concern|risk|safety|"
        r"appropriat(?:e|eness)|inappropriat(?:e|eness)|"
        r"contraindicat(?:e|ed|es|ion))\b",
        rf"\b{CROSS_PATIENT_SOURCE}\b.{{0,180}}\b"
        r"(?:frequently|commonly|often)\s+(?:used|indicated|included)\b",
        r"\b(?:contradict(?:s|ed|ing)?|corroborat(?:e|es|ed|ing)|"
        r"demonstrat(?:e|es|ed|ing)|indicat(?:e|es|ed|ing)|"
        r"prov(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|support(?:s|ed|ing)?)\b"
        r".{0,120}\b"
        r"(?:us(?:e|ed|es|ing)|inclusion|exclusion|probability|"
        r"indicat(?:e|ed|es|ing|ion)|concern|risk|safety|"
        r"appropriat(?:e|eness)|inappropriat(?:e|eness)|"
        r"contraindicat(?:e|ed|es|ion))\b"
        rf".{{0,180}}\b{CROSS_PATIENT_SOURCE}\b",
        rf"\b{CROSS_PATIENT_SOURCE}\b.{{0,180}}\b"
        r"(?:preferred?|preference|favou?red?|alternative)\b",
    )
)

RETRIEVER_FINAL_RECOMMENDATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:therefore|without these details|for this patient|the patient)"
        r"\b.{0,220}\b(?:should|must|needs? to)\b.{0,100}\b"
        r"(?:avoid(?:ed|ing)?|continu(?:e|ed|ing)|"
        r"discontinu(?:e|ed|ing)|held|hold|initiat(?:e|ed|ing)|"
        r"prescrib(?:e|ed|ing)|receiv(?:e|ed|ing)|resum(?:e|ed|ing)|"
        r"start(?:ed|ing)?|stop(?:ped|ping)?|switch(?:ed|ing)?|"
        r"tak(?:e|en|ing)|us(?:e|ed|ing)|withheld|withhold)\b",
        r"\balternative\b.{0,100}\b(?:agents?|drugs?|medications?|"
        r"strateg(?:y|ies)|therap(?:y|ies))\b.{0,100}\b"
        r"should be considered\b",
        r"\bnecessitat(?:e|es|ed|ing)\b.{0,100}\b"
        r"(?:avoidance|discontinuation|initiation|prescribing|withholding)\b",
        r"\b(?:current\s+)?standard of care\b.{0,100}\b"
        r"(?:avoid(?:ance|ed|ing)?|discontinu(?:ation|e|ed|ing)|"
        r"hold|withhold|withholding|stop|stopping)\b",
        r"\b(?:therefore|thus|overall)\b.{0,220}\b"
        r"(?:making (?:it|the medication)\s+)?"
        r"(?:inappropriate|unsuitable|preferred|the safer choice)\b",
        r"\b(?:(?:making (?:it|the medication)\s+)"
        r"(?:inappropriate|unsuitable)|the safer choice)\b",
    )
)

CONDITIONAL_EVIDENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bconditional\b",
        r"\bdepends? on\b",
        r"\b(?:may|might|could)\b",
        r"\b(?:related to|var(?:y|ies|ied|ying) with)\b.{0,80}\b"
        r"(?:severity|reversibility|organ function|exposure)\b",
        r"\binformation (?:is )?still needed\b",
        r"\b(?:current|baseline|serial)\b.{0,50}\b"
        r"(?:dose|eGFR|laboratory|organ function|severity|exposure)\b",
        r"\b(?:dose|eGFR|laboratory|organ function|severity|exposure)\b"
        r".{0,80}\b(?:needed|required|unknown|unavailable|missing)\b",
        r"\b(?:lack(?:s|ed|ing)?|limited|no)\b.{0,50}\b"
        r"(?:direct (?:stud(?:y|ies)|evidence)|randomi[sz]ed|RCTs?)\b",
        r"\b(?:uncertain|unstable|"
        r"reassess(?:ed|es|ing|ment)?|"
        r"re-evaluat(?:e|ed|es|ing|ion))\b",
        r"\beGFR\b.{0,40}(?:[<>]=?|above|below|greater than|less than|"
        r"threshold|\d+(?:\.\d+)?)",
    )
)

ATTRIBUTED_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:FDA|EMA|MHRA|label(?:ing)?|product label(?:ing)?|guideline(?:s)?|"
    r"consensus|societ(?:y|ies)|manufacturer|evidence)\b.{0,100}\b"
    r"(?:advis(?:e|es|ed)|contraindicat(?:e|es|ed)|recommend(?:s|ed)?|"
    r"state(?:s|d)?|warn(?:s|ed)?)\b",
    re.IGNORECASE,
)

GENERIC_PRESCRIBING_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:should|must|needs? to)\b.{0,100}\b"
    r"(?:avoid(?:ed|ing)?|continu(?:e|ed|ing)|"
    r"discontinu(?:e|ed|ing)|held|hold|initiat(?:e|ed|ing)|"
    r"prescrib(?:e|ed|ing)|receiv(?:e|ed|ing)|resum(?:e|ed|ing)|"
    r"start(?:ed|ing)?|stop(?:ped|ping)?|switch(?:ed|ing)?|"
    r"tak(?:e|en|ing)|us(?:e|ed|es|ing)|withheld|withhold)\b",
    re.IGNORECASE,
)

UNQUALIFIED_CATEGORICAL_PATTERN = re.compile(
    r"\b(?:always|clearly|contraindicat(?:e|ed|es|ion)|never|regardless|"
    r"necessitat(?:e|es|ed|ing)|should be "
    r"(?:avoided|discontinued|held|withheld)|"
    r"(?:avoid|discontinue|withhold|stop)\s+[A-Z][\w-]*)\b",
    re.IGNORECASE,
)

GENERIC_ALTERNATIVE_PATTERN = re.compile(
    r"\b(?:alternative|other)\s+"
    r"(?:agents?|drugs?|medications?|therap(?:y|ies))\b",
    re.IGNORECASE,
)

EXPLICIT_CLINICAL_QUALIFIER_PATTERN = re.compile(
    r"\b(?:above|below|greater than|if|less than|when)\b"
    r".{0,80}\b(?:dose|eGFR|laboratory|organ function|severity|exposure)\b"
    r"|\b(?:dose|eGFR|laboratory|organ function|severity|exposure)\b"
    r".{0,80}\b(?:above|below|greater than|less than|threshold|[<>]=?)\b",
    re.IGNORECASE,
)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def medication_prediction_names(sample: ResearchSample) -> tuple[str, ...]:
    return tuple(item.name for item in sample.medication_candidates)


def mentions_name(text: str, name: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(name)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
    )


def visit_medication_names(sample: ResearchSample) -> set[str]:
    names: set[str] = set()
    for visit in (*sample.history, *sample.similar_visits):
        names.update(
            str(name).strip()
            for name in visit.get("medications", [])
            if str(name).strip()
        )
    return names


def context_medication_names(sample: ResearchSample) -> set[str]:
    names = set(medication_prediction_names(sample))
    names.update(visit_medication_names(sample))
    for pair in sample.ddi_constraints:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            names.update(str(name).strip() for name in pair)
    return {name for name in names if name}


def reporter_allowed_medication_names(sample: ResearchSample) -> set[str]:
    names = {
        item.name
        for item in (
            sample.boundary_medications + sample.confident_accept
        )
    }
    for pair in sample.ddi_constraints:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            names.update(str(name).strip() for name in pair)
    return {name for name in names if name}


def foreign_reporter_medication_names(
    sample: ResearchSample,
) -> set[str]:
    allowed = {
        name.casefold()
        for name in reporter_allowed_medication_names(sample)
    }
    return {
        name
        for name in visit_medication_names(sample)
        if name.casefold() not in allowed
    }


def joined_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def generic_evidence_objective(
    affected_candidates: tuple[str, ...],
) -> str:
    return (
        "Clarify the patient-specific indication, safety, transferability, "
        "or evidence-gap considerations for "
        f"{joined_names(affected_candidates)} under the stated contrast."
    )


def objective_allowed_medication_names(
    sample: ResearchSample,
    affected_candidates: tuple[str, ...],
) -> set[str]:
    allowed = set(affected_candidates)
    affected = {name.casefold() for name in affected_candidates}
    for pair in sample.ddi_constraints:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        first = str(pair[0]).strip()
        second = str(pair[1]).strip()
        if first.casefold() in affected:
            allowed.add(second)
        if second.casefold() in affected:
            allowed.add(first)
    return {name for name in allowed if name}


def repair_planner_foreign_medication_context(
    text: str,
    sample: ResearchSample,
    affected_candidates: tuple[str, ...],
) -> tuple[str, list[str]]:
    """Keep cross-patient medication patterns contrastive and unnamed."""
    repaired = normalized_text(text)
    allowed = {
        name.casefold()
        for name in objective_allowed_medication_names(
            sample,
            affected_candidates,
        )
    }
    removed: list[str] = []
    for name in sorted(context_medication_names(sample), key=len, reverse=True):
        if name.casefold() in allowed or not mentions_name(repaired, name):
            continue
        escaped = re.escape(name)
        replacements = (
            (
                rf"\b(?:the\s+)?(?:use|prescription|administration)\s+of\s+"
                rf"{escaped}\b",
                "a different cross-patient medication pattern",
            ),
            (
                rf"\b{escaped}\s+(?:use|therapy|treatment|prescription)\b",
                "a different cross-patient medication pattern",
            ),
            (
                rf"(?<!\w){escaped}(?!\w)",
                "a different cross-patient medication",
            ),
        )
        for pattern, replacement in replacements:
            repaired, count = re.subn(
                pattern,
                replacement,
                repaired,
                flags=re.IGNORECASE,
            )
            if count:
                removed.append(name)
                break
    repairs = []
    if removed:
        repairs.append(
            "Reframed non-objective cross-patient medication names as an "
            "unnamed contrastive medication pattern: "
            + ", ".join(sorted(set(removed)))
            + "."
        )
    return normalized_text(repaired), repairs


def repair_planner_exposure_language(
    text: str,
    sample: ResearchSample,
) -> tuple[str, list[str]]:
    repaired = text
    repairs: list[str] = []
    repaired, generic_absence_count = re.subn(
        r"\b(?:the\s+)?current absence of\s+"
        r"(?:a|an|any|the)?\s*(?:specific\s+)?"
        r"(anticoagulant|drug|medication|therapy|treatment)\b",
        (
            r"the fact that the current input does not establish observed "
            r"\1 exposure"
        ),
        repaired,
        flags=re.IGNORECASE,
    )
    if generic_absence_count:
        repairs.append(
            "Reframed an inferred current medication-class absence."
        )
    current_context = (
        r"(?:the\s+)?current(?:\s+patient)?"
        r"(?:\s+state|\s+visit|\s+medication(?:s|\s+list|\s+regimen)?)"
    )
    for name in medication_prediction_names(sample):
        escaped_name = re.escape(name)
        quoted_name = (
            rf"(?<!\w)['\"]?{escaped_name}['\"]?(?!\w)"
        )
        no_current_medication_pattern = (
            rf"\b(?:the\s+)?current patient\s+has\s+no\s+{quoted_name}"
            r"\s+in\s+(?:its\s+)?medications\b"
        )
        repaired, no_current_count = re.subn(
            no_current_medication_pattern,
            (
                "the current input provides no observed medication exposure "
                f"for {name}"
            ),
            repaired,
            flags=re.IGNORECASE,
        )
        if no_current_count:
            repairs.append(
                f"Reframed inferred current exposure for {name}."
            )
        source_pattern = (
            rf"\b{current_context}\s+(?:do|does)\s+not\s+include\s+"
            rf"{quoted_name}\s*,\s*which was present in\s+([^.;]+)"
        )
        repaired, source_count = re.subn(
            source_pattern,
            (
                f"{name} was recorded in "
                r"\1; the current input provides no observed medication "
                "exposure"
            ),
            repaired,
            flags=re.IGNORECASE,
        )
        if source_count:
            repairs.append(
                f"Reframed inferred current exposure for {name}."
            )
        replacement = (
            f"{name} is a model-predicted candidate rather than an "
            "observed current medication exposure"
        )
        patterns = (
            rf"\b(?:the\s+)?absence of\s+{quoted_name}\s+in\s+"
            rf"{current_context}",
            rf"\b{current_context}\s+(?:do|does)\s+not\s+include\s+"
            rf"{quoted_name}",
            rf"{quoted_name}\s+(?:is|was)\s+(?:absent|missing)\s+"
            rf"(?:from|in)\s+{current_context}",
        )
        for pattern in patterns:
            repaired, count = re.subn(
                pattern,
                replacement,
                repaired,
                flags=re.IGNORECASE,
            )
            if count:
                repairs.append(
                    f"Reframed inferred current exposure for {name}."
                )
        candidate_relevance = (
            f"the potential relevance of {name} to the current patient"
        )
        exposure_patterns = (
            rf"\b{quoted_name}\s+(?:is|was)\s+(?:currently\s+)?"
            rf"(?:prescribed|administered|given|received|taken|used)\s+"
            rf"(?:in|by|for|to)\s+(?:the\s+)?current patient\b",
            rf"\b(?:the\s+)?current patient['\u2019]s\s+"
            rf"(?:use|receipt|administration)\s+of\s+{quoted_name}",
            rf"\b{quoted_name}\s+use\s+in\s+(?:a|the)\s+patient\b",
            rf"\b{quoted_name}\s+use\s+in\s+the\s+context\b",
        )
        for pattern in exposure_patterns:
            repaired, count = re.subn(
                pattern,
                candidate_relevance,
                repaired,
                flags=re.IGNORECASE,
            )
            if count:
                repairs.append(
                    f"Reframed inferred current exposure for {name} as "
                    "candidate relevance."
                )
    return normalized_text(repaired), repairs


def repair_planner_research_objective(
    text: str,
    affected_candidates: tuple[str, ...],
) -> tuple[str, list[str]]:
    normalized = normalized_text(text)
    prescribing_decision = any(
        pattern.search(normalized)
        for pattern in PRESCRIBING_DECISION_PATTERNS
    )
    retrieval_question = bool(
        normalized.endswith("?")
        or re.match(
            r"^(?:what|when|where|which|who|why|how|does|do|is|are|"
            r"can|could|should|would)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if not prescribing_decision and not retrieval_question:
        return normalized, []
    if retrieval_question and not prescribing_decision:
        if DDI_PATTERN.search(normalized):
            return (
                "Clarify the interaction mechanism and patient-specific "
                "safety evidence for "
                f"{joined_names(affected_candidates)} under the supplied "
                "DDI constraint without assuming observed co-exposure.",
                [
                    "Reframed a Planner DDI question as a declarative "
                    "research objective."
                ],
            )
        clinical_evidence = re.match(
            r"^what\s+is\s+the\s+clinical\s+evidence\s+regarding\s+"
            r"(.+?)\??$",
            normalized,
            flags=re.IGNORECASE,
        )
        if clinical_evidence:
            subject = clinical_evidence.group(1).rstrip(" ?.")
            return (
                f"Clarify the clinical evidence regarding {subject}.",
                [
                    "Reframed a Planner retrieval question as a declarative "
                    "research objective."
                ],
            )
    return (
        generic_evidence_objective(affected_candidates),
        [
            "Reframed a Planner retrieval question or prescribing decision "
            "as a declarative evidence objective."
        ],
    )


def repair_planner_ddi_objective(
    text: str,
    sample: ResearchSample,
    affected_candidates: tuple[str, ...],
) -> tuple[str, list[str]]:
    normalized = normalized_text(text)
    if not DDI_PATTERN.search(normalized):
        return normalized, []
    mentioned = {
        name
        for name in medication_prediction_names(sample)
        if name.casefold() in normalized.casefold()
    }
    grounded = any(
        str(pair[0]) in mentioned and str(pair[1]) in mentioned
        for pair in sample.ddi_constraints
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )
    if grounded:
        return normalized, []
    return (
        generic_evidence_objective(affected_candidates),
        ["Removed a DDI claim not grounded in the supplied constraints."],
    )


def repair_planner_ddi_exposure_language(
    text: str,
    sample: ResearchSample,
    affected_candidates: tuple[str, ...],
) -> tuple[str, list[str]]:
    """Express grounded DDI research without assuming observed exposure."""
    normalized = normalized_text(text)
    affected = {name.casefold() for name in affected_candidates}
    for pair in sample.ddi_constraints:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        first = str(pair[0]).strip()
        second = str(pair[1]).strip()
        if not (
            {first.casefold(), second.casefold()} & affected
            and mentions_name(normalized, first)
            and mentions_name(normalized, second)
        ):
            continue
        if not re.search(
            r"\b(?:co-?administ(?:er|ered|ering|ration)|"
            r"concurrent(?:ly)?\s+us(?:e|ed|ing)|"
            r"used?\s+(?:together|with))\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            continue
        return (
            "Clarify the interaction mechanism and patient-specific risk "
            f"evidence for conditional co-recommendation of {first} and "
            f"{second}, including the clinical factors needed to assess the "
            "known DDI without assuming observed medication exposure.",
            [
                "Reframed a grounded DDI objective as conditional "
                "co-recommendation rather than observed exposure."
            ],
        )
    return normalized, []


def repair_planner_monitoring_objective(
    text: str,
    affected_candidates: tuple[str, ...],
) -> tuple[str, list[str]]:
    normalized = normalized_text(text)
    lowered = normalized.casefold()
    if "monitoring" not in lowered:
        return normalized, []
    causal_terms = (
        "affect",
        "appropriateness",
        "clinical outcome",
        "effectiveness",
        "impact",
        "influence",
        "use of",
    )
    if not any(term in lowered for term in causal_terms):
        return normalized, []
    return (
        "Clarify the patient-specific indication and safety evidence for "
        f"{joined_names(affected_candidates)}, including which organ-function, "
        "laboratory, dose, exposure, or disease-severity information is needed "
        "under the stated contrast.",
        ["Reframed monitoring as information rather than a causal exposure."],
    )


def validate_grounded_ddi_claim(
    text: str,
    sample: ResearchSample,
    *,
    context: str,
) -> None:
    normalized = normalized_text(text)
    if not DDI_PATTERN.search(normalized):
        return
    mentioned = {
        name
        for name in medication_prediction_names(sample)
        if name.casefold() in normalized.casefold()
    }
    if any(
        str(pair[0]) in mentioned and str(pair[1]) in mentioned
        for pair in sample.ddi_constraints
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ):
        return
    raise ValueError(
        f"{context} introduced a DDI claim not grounded in the supplied "
        "DDI constraints."
    )


def validate_no_inferred_current_exposure(
    text: str,
    sample: ResearchSample,
    *,
    context: str,
) -> None:
    normalized = normalized_text(text)
    if not any(
        name.casefold() in normalized.casefold()
        for name in medication_prediction_names(sample)
    ):
        return
    current_context = (
        r"(?:the\s+)?current(?:\s+patient)?"
        r"(?:\s+state|\s+visit|\s+medication(?:s|\s+list|\s+regimen)?)"
    )
    # State differences such as "without exacerbation" are valid Planner
    # contrasts. Treat negation as medication non-exposure only when the
    # negated phrase is explicitly tied to a supplied candidate name.
    named_state_exposure = any(
        any(
            re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )
            for pattern in (
                rf"\babsence of\s+['\"]?{re.escape(name)}['\"]?\b"
                rf".{{0,80}}\b{current_context}\b",
                rf"\b{current_context}\b.{{0,100}}\b"
                rf"(?:absent|absence|missing|does not include|do not "
                rf"include|not include|without)\b\s+"
                rf"(?:(?:any|the|a|an)\s+)?(?:boundary\s+)?"
                rf"(?:(?:candidate|medication|drug|therapy|treatment)\s+)?"
                rf"['\"]?{re.escape(name)}['\"]?\b",
                rf"\b['\"]?{re.escape(name)}['\"]?\b.{{0,50}}\b"
                rf"(?:absent|missing)\b.{{0,50}}\b(?:from|in)\s+"
                rf"{current_context}\b",
            )
        )
        for name in medication_prediction_names(sample)
    )
    named_exposure = any(
        re.search(
            r"\bpatients?\b.{0,120}\b(?:who\s+)?"
            r"(?:(?:is|was)\s+)?(?:currently\s+)?"
            rf"(?:on|taking|receiving|using)\s+{re.escape(name)}\b",
            normalized,
            flags=re.IGNORECASE,
        )
        for name in medication_prediction_names(sample)
    )
    if (
        named_state_exposure
        or named_exposure
        or any(
            pattern.search(normalized)
            for pattern in CURRENT_EXPOSURE_PATTERNS
        )
    ):
        raise ValueError(
            f"{context} inferred current medication exposure or non-exposure "
            "from model predictions. Describe them only as predicted candidates "
            "or use conditional co-recommendation wording."
        )


def repair_question_exposure_language(
    question: str,
    sample: ResearchSample,
) -> tuple[str, list[str]]:
    """Express predicted medication context without claiming exposure."""
    repaired = normalized_text(question)
    repairs: list[str] = []
    for name in medication_prediction_names(sample):
        escaped = re.escape(name)
        patterns = (
            rf"\bwho\s+(?:is|are|was|were)\s+"
            rf"(?:currently\s+)?on\s+{escaped}\b",
            rf"\b(?:currently\s+)?"
            rf"(?:taking|receiving|using)\s+{escaped}\b",
        )
        replacement = (
            f"for whom {name} is a predicted conditional "
            "co-recommendation"
        )
        for pattern in patterns:
            repaired, count = re.subn(
                pattern,
                replacement,
                repaired,
                flags=re.IGNORECASE,
            )
            if count:
                repairs.append(
                    f"Reframed inferred current exposure to {name} as a "
                    "conditional co-recommendation."
                )
    return normalized_text(repaired), repairs


def split_sentences(text: str) -> list[str]:
    normalized = normalized_text(text)
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            normalized,
        )
        if sentence.strip()
    ]


def repair_retriever_answer(
    answer: str,
) -> tuple[str, list[str]]:
    """Remove direct prescribing conclusions from Retriever evidence."""
    retained: list[str] = []
    repairs: list[str] = []
    answer_is_conditional = any(
        pattern.search(answer)
        for pattern in CONDITIONAL_EVIDENCE_PATTERNS
    )
    for sentence in split_sentences(answer):
        attributed_recommendation = bool(
            ATTRIBUTED_RECOMMENDATION_PATTERN.search(sentence)
        )
        direct_recommendation = (
            any(
                pattern.search(sentence)
                for pattern in RETRIEVER_FINAL_RECOMMENDATION_PATTERNS
            )
            or bool(
                GENERIC_PRESCRIBING_RECOMMENDATION_PATTERN.search(sentence)
            )
        ) and not attributed_recommendation
        rejects_information_need = bool(
            re.search(
                r"\bno (?:further|additional|other) information\b"
                r".{0,50}\b(?:needed|required)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        context_free_absolute = bool(
            re.search(
                r"\bregardless of\b.{0,80}\b"
                r"(?:severity|factor(?:s)?|context|organ function|eGFR)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        unrelated_alternative = bool(
            re.search(
                r"\balternative\b.{0,80}\b"
                r"(?:agents?|drugs?|medications?|strateg(?:y|ies)|"
                r"therap(?:y|ies))\b",
                sentence,
                flags=re.IGNORECASE,
            )
        )
        contradictory_absolute = (
            answer_is_conditional
            and bool(UNQUALIFIED_CATEGORICAL_PATTERN.search(sentence))
            and not bool(EXPLICIT_CLINICAL_QUALIFIER_PATTERN.search(sentence))
        )
        if direct_recommendation:
            repairs.append(
                "Removed a patient-level prescribing conclusion from "
                "Retriever evidence."
            )
            continue
        if rejects_information_need or context_free_absolute:
            repairs.append(
                "Removed a context-free Retriever assertion that rejected "
                "patient-specific evidence requirements."
            )
            continue
        if unrelated_alternative:
            repairs.append(
                "Removed an unrelated alternative-medication statement from "
                "Retriever evidence."
            )
            continue
        if contradictory_absolute:
            repairs.append(
                "Removed an absolute Retriever claim that conflicted with "
                "its stated patient-specific evidence requirements."
            )
            continue
        retained.append(sentence)
    repaired = normalized_text(" ".join(retained))
    if not repaired:
        repaired = (
            "No admissible direct clinical evidence remained after removing "
            "patient-level prescribing conclusions from the retrieved answer."
        )
        repairs.append(
            "Converted an evidence-free Retriever response into an explicit "
            "evidence gap."
        )
    return repaired, repairs


def direct_answers_by_medication(
    plan: PersonalizedPlan,
    memory: list[EvidenceItem],
) -> dict[str, list[str]]:
    answers = {
        medication: []
        for objective in plan.research_objectives
        for medication in objective.affected_candidates
    }
    objectives = {
        objective.objective_id: objective
        for objective in plan.research_objectives
    }
    for item in memory:
        answer = normalized_text(item.answer or "")
        objective = objectives.get(item.objective_id)
        if (
            not answer
            or answer.startswith(
                "No admissible direct clinical evidence remained"
            )
            or objective is None
        ):
            continue
        question_and_answer = normalized_text(
            f"{item.question} {answer}"
        ).casefold()
        for medication in objective.affected_candidates:
            if medication.casefold() in question_and_answer:
                answers.setdefault(medication, []).append(answer)
    return answers


def direct_evidence_excerpt(
    medication: str,
    answers: list[str],
) -> str:
    sentences = [
        sentence
        for answer in answers
        for sentence in split_sentences(answer)
    ]
    selected = [
        sentence
        for sentence in sentences
        if medication.casefold() in sentence.casefold()
    ]
    if not selected:
        selected = sentences
    return normalized_text(" ".join(selected[:2]))[:800]


def evidence_is_conditional(answers: list[str]) -> bool:
    combined = normalized_text(" ".join(answers))
    return any(
        pattern.search(combined)
        for pattern in CONDITIONAL_EVIDENCE_PATTERNS
    )


def remove_sentences_with_names(
    text: str,
    names: set[str],
) -> tuple[str, set[str]]:
    removed: set[str] = set()
    retained: list[str] = []
    for sentence in split_sentences(text):
        mentioned = {
            name for name in names if mentions_name(sentence, name)
        }
        if mentioned:
            removed.update(mentioned)
        else:
            retained.append(sentence)
    return normalized_text(" ".join(retained)), removed


def remove_indirect_proof_sentences(text: str) -> tuple[str, bool]:
    retained: list[str] = []
    removed = False
    for sentence in split_sentences(text):
        if any(
            pattern.search(sentence)
            for pattern in (
                *REPORTER_MONITORING_PROOF_PATTERNS,
                *REPORTER_CROSS_PATIENT_PROOF_PATTERNS,
            )
        ):
            removed = True
        else:
            retained.append(sentence)
    return normalized_text(" ".join(retained)), removed


def remove_generic_alternative_sentences(
    text: str,
) -> tuple[str, bool]:
    retained: list[str] = []
    removed = False
    for sentence in split_sentences(text):
        if GENERIC_ALTERNATIVE_PATTERN.search(sentence):
            removed = True
        else:
            retained.append(sentence)
    return normalized_text(" ".join(retained)), removed


def reporter_direct_evidence_excerpt(
    medication: str,
    answers: list[str],
    foreign_names: set[str],
) -> str:
    excerpt = direct_evidence_excerpt(medication, answers)
    excerpt, _ = remove_sentences_with_names(excerpt, foreign_names)
    return excerpt or (
        "The objective-linked retrieval contains direct evidence, but its "
        "medication-comparison context is outside the Reporter candidate set."
    )


def repair_reporter_candidate_exposure_language(
    text: str,
    sample: ResearchSample,
) -> tuple[str, list[str]]:
    """Distinguish evaluated boundary candidates from observed therapy."""
    repaired = normalized_text(text)
    repairs: list[str] = []
    repaired, plan_count = re.subn(
        r"\b(?:the\s+)?current\s+(?:treatment|medication)\s+plan\s+"
        r"(?:includes?|involves?|contains?)\b",
        "The boundary-candidate evaluation includes",
        repaired,
        flags=re.IGNORECASE,
    )
    if plan_count:
        repairs.append(
            "Reframed the current treatment plan as a boundary-candidate "
            "evaluation."
        )
    for medication in sample.boundary_medications:
        name = medication.name
        escaped = re.escape(name)
        replacement = f"the evaluated relevance of {name} for the patient"
        patterns = (
            rf"\b(?:the\s+)?current patient['\u2019]s\s+"
            rf"(?:use|receipt|administration)\s+of\s+{escaped}\b",
            rf"\b(?:the\s+)?current patient\s+(?:is|was)\s+"
            rf"(?:currently\s+)?(?:on|taking|receiving|using)\s+{escaped}\b",
            rf"\b{escaped}\s+(?:is|was)\s+(?:currently\s+)?prescribed\s+"
            rf"(?:in|for|to)\s+(?:the\s+)?current patient\b",
        )
        for pattern in patterns:
            repaired, count = re.subn(
                pattern,
                replacement,
                repaired,
                flags=re.IGNORECASE,
            )
            if count:
                repairs.append(
                    f"Reframed inferred current exposure for {name} in the "
                    "Reporter output."
                )
    return normalized_text(repaired), repairs


def repair_reporter_context_as_proof(
    report: dict[str, Any],
    sample: ResearchSample,
    plan: PersonalizedPlan,
    memory: list[EvidenceItem],
    *,
    max_conditional_shift: float = REPORTER_MAX_CONDITIONAL_SHIFT,
) -> tuple[dict[str, Any], list[str]]:
    """Retain the prior when a stated update relies on contrastive context.

    The paper permits the Reporter to use monitoring procedures and
    cross-patient visits as context, but not as target-patient clinical proof.
    This post-parse calibration preserves direct objective-linked evidence,
    removes unsupported or foreign medication context, and limits excessive
    probability shifts when the direct evidence remains conditional.
    """
    if not 0.0 < max_conditional_shift <= 1.0:
        raise ValueError("Reporter conditional shift limit must be in (0, 1].")
    repairs: list[str] = []
    direct_answers = direct_answers_by_medication(plan, memory)
    foreign_names = foreign_reporter_medication_names(sample)
    medications: list[dict[str, Any]] = []
    for original in report["medications"]:
        row = dict(original)
        base = float(row["base_probability"])
        adjusted = float(row["adjusted_probability"])
        medication = str(row["name"])
        medication_answers = direct_answers.get(medication, [])
        if not medication_answers:
            row["adjusted_probability"] = base
            row["evidence"] = (
                f"No direct objective-linked retrieval was selected for "
                f"{medication}."
            )
            row["reason"] = (
                "The available plan and longitudinal or cross-patient "
                "contexts are indirect, so the frozen base probability is "
                "retained."
            )
            repairs.append(
                f"Retained the base probability for {medication} because no "
                "direct medication-linked retrieval was available."
            )
            medications.append(row)
            continue

        reason, reason_exposure_repairs = (
            repair_reporter_candidate_exposure_language(
                str(row["reason"]),
                sample,
            )
        )
        evidence, evidence_exposure_repairs = (
            repair_reporter_candidate_exposure_language(
                str(row["evidence"]),
                sample,
            )
        )
        repairs.extend(reason_exposure_repairs)
        repairs.extend(evidence_exposure_repairs)
        evidence_key = evidence.casefold().rstrip(".")
        if (
            evidence_key
            in {
                "accepted evidence",
                "direct evidence",
                "evidence available",
                "no direct retrieval",
                "no direct evidence",
            }
            or evidence_key.startswith("no direct objective-linked retrieval")
        ):
            row["evidence"] = (
                f"Direct objective-linked retrieval for {medication}: "
                + reporter_direct_evidence_excerpt(
                    medication,
                    medication_answers,
                    foreign_names,
                )
            )
            evidence = normalized_text(str(row["evidence"]))
            repairs.append(
                f"Replaced a generic or inconsistent evidence label for "
                f"{medication} with its objective-linked retrieval."
            )
        reason, foreign_in_reason = remove_sentences_with_names(
            reason,
            foreign_names,
        )
        evidence, foreign_in_evidence = remove_sentences_with_names(
            evidence,
            foreign_names,
        )
        reason, alternative_in_reason = (
            remove_generic_alternative_sentences(reason)
        )
        evidence, alternative_in_evidence = (
            remove_generic_alternative_sentences(evidence)
        )
        if (
            foreign_in_reason
            or foreign_in_evidence
            or alternative_in_reason
            or alternative_in_evidence
        ):
            evidence = (
                f"Direct objective-linked retrieval for {medication}: "
                + reporter_direct_evidence_excerpt(
                    medication,
                    medication_answers,
                    foreign_names,
                )
            )
            reason = (
                "The calibrated change uses only the current objective and "
                "its direct retrieval; unrelated medication context is not "
                "treated as target-patient evidence."
            )
            repairs.append(
                f"Removed medication context outside the Reporter candidate "
                f"set for {medication}: "
                + ", ".join(
                    sorted(foreign_in_reason | foreign_in_evidence)
                )
                + "."
            )
        if alternative_in_reason or alternative_in_evidence:
            repairs.append(
                f"Removed unspecified alternative-medication context from "
                f"the {medication} explanation."
            )
        row["reason"] = reason
        row["evidence"] = evidence
        monitoring_as_proof = any(
            pattern.search(reason)
            for pattern in REPORTER_MONITORING_PROOF_PATTERNS
        )
        cross_patient_as_proof = any(
            pattern.search(reason)
            for pattern in REPORTER_CROSS_PATIENT_PROOF_PATTERNS
        )
        if abs(adjusted - base) > 1e-12 and (
            monitoring_as_proof or cross_patient_as_proof
        ):
            sources: list[str] = []
            evidence_gaps: list[str] = []
            if monitoring_as_proof:
                sources.append("monitoring procedure")
                evidence_gaps.append(
                    "the procedure name does not supply the corresponding "
                    "target-patient laboratory or organ-function result"
                )
            if cross_patient_as_proof:
                sources.append("cross-patient pattern")
                evidence_gaps.append(
                    "similar-visit medication patterns are contrastive rather "
                    "than target-patient clinical proof"
                )
            row["adjusted_probability"] = base
            row["evidence"] = (
                "The stated update relies on "
                + " and ".join(sources)
                + "; "
                + " and ".join(evidence_gaps)
                + "."
            )
            row["reason"] = (
                "Because the stated probability change relies on indirect "
                "context rather than the direct objective-linked retrieval, "
                "the frozen base probability is retained."
            )
            repairs.append(
                "Retained the base probability for "
                f"{medication} because the Reporter treated "
                + " and ".join(sources)
                + " as clinical proof."
            )
            medications.append(row)
            continue

        evidence_uses_indirect_proof = any(
            pattern.search(evidence)
            for pattern in (
                *REPORTER_MONITORING_PROOF_PATTERNS,
                *REPORTER_CROSS_PATIENT_PROOF_PATTERNS,
            )
        )
        if evidence_uses_indirect_proof:
            row["evidence"] = (
                f"Direct objective-linked retrieval for {medication}: "
                + reporter_direct_evidence_excerpt(
                    medication,
                    medication_answers,
                    foreign_names,
                )
            )
            repairs.append(
                f"Removed indirect contrastive proof from the {medication} "
                "evidence statement."
            )

        adjusted = float(row["adjusted_probability"])
        shift = adjusted - base
        conditional_evidence = evidence_is_conditional(
            medication_answers
        )
        if conditional_evidence and (
            UNQUALIFIED_CATEGORICAL_PATTERN.search(evidence)
            or UNQUALIFIED_CATEGORICAL_PATTERN.search(reason)
        ):
            row["evidence"] = (
                f"Direct objective-linked retrieval for {medication}: "
                + reporter_direct_evidence_excerpt(
                    medication,
                    medication_answers,
                    foreign_names,
                )
            )
            row["reason"] = (
                "The direct retrieval identifies a clinically relevant "
                "mechanism but also states that interpretation depends on "
                "missing patient-specific information; the calibrated change "
                "reflects both."
            )
            repairs.append(
                f"Removed an unqualified categorical Reporter explanation "
                f"for {medication} because the direct evidence is conditional."
            )
        if (
            abs(shift) > max_conditional_shift
            and conditional_evidence
        ):
            calibrated = (
                base + max_conditional_shift
                if shift > 0.0
                else base - max_conditional_shift
            )
            row["adjusted_probability"] = min(max(calibrated, 0.0), 1.0)
            repairs.append(
                f"Limited the conditional-evidence probability shift for "
                f"{medication} to {max_conditional_shift:.3f}."
            )
        medications.append(row)

    repaired = dict(report)
    repaired["medications"] = medications
    clinical_summary, summary_exposure_repairs = (
        repair_reporter_candidate_exposure_language(
            str(report["clinical_summary"]),
            sample,
        )
    )
    repairs.extend(summary_exposure_repairs)
    clinical_summary, summary_foreign = remove_sentences_with_names(
        clinical_summary,
        foreign_names,
    )
    clinical_summary, summary_indirect = remove_indirect_proof_sentences(
        clinical_summary
    )
    clinical_summary, summary_alternative = (
        remove_generic_alternative_sentences(clinical_summary)
    )
    if not clinical_summary:
        clinical_summary = (
            "The report updates boundary-medication probabilities only from "
            "direct objective-linked retrieval and retains uncertainty when "
            "patient-specific evidence is incomplete."
        )
    if summary_foreign:
        repairs.append(
            "Removed medication context outside the Reporter candidate set "
            "from the clinical summary: "
            + ", ".join(sorted(summary_foreign))
            + "."
        )
    if summary_indirect:
        repairs.append(
            "Removed indirect contrastive proof from the clinical summary."
        )
    if summary_alternative:
        repairs.append(
            "Removed unspecified alternative-medication context from the "
            "clinical summary."
        )
    repaired["clinical_summary"] = clinical_summary

    safety_alerts: list[str] = []
    for raw_alert in report["safety_alerts"]:
        alert, alert_exposure_repairs = (
            repair_reporter_candidate_exposure_language(
                str(raw_alert),
                sample,
            )
        )
        repairs.extend(alert_exposure_repairs)
        alert, alert_foreign = remove_sentences_with_names(
            alert,
            foreign_names,
        )
        if alert_foreign:
            repairs.append(
                "Removed medication context outside the Reporter candidate "
                "set from a safety alert: "
                + ", ".join(sorted(alert_foreign))
                + "."
            )
        alert, alert_indirect = remove_indirect_proof_sentences(alert)
        if alert_indirect:
            repairs.append(
                "Removed indirect contrastive proof from a safety alert."
            )
        alert_has_alternative = bool(
            GENERIC_ALTERNATIVE_PATTERN.search(alert)
        )
        conditional_medications = [
            medication
            for medication, answers in direct_answers.items()
            if (
                mentions_name(alert, medication)
                and evidence_is_conditional(answers)
            )
        ]
        if (
            alert
            and conditional_medications
            and (
                alert_has_alternative
                or (
                    UNQUALIFIED_CATEGORICAL_PATTERN.search(alert)
                    and not EXPLICIT_CLINICAL_QUALIFIER_PATTERN.search(
                        alert
                    )
                )
            )
        ):
            medication = conditional_medications[0]
            alert = (
                f"For {medication}, apply the retrieved safety evidence "
                "conditionally and verify the required laboratory, "
                "organ-function, dose, exposure, and disease-severity "
                "information."
            )
            repairs.append(
                f"Reframed an unqualified safety alert for {medication} "
                "because the direct evidence is conditional."
            )
        elif alert_has_alternative:
            alert = ""
            repairs.append(
                "Removed unspecified alternative-medication context from a "
                "safety alert."
            )
        if alert:
            safety_alerts.append(alert)
    repaired["safety_alerts"] = safety_alerts
    return repaired, repairs


def validate_planner_objective(
    objective: ResearchObjective,
    sample: ResearchSample,
) -> None:
    combined = normalized_text(
        f"{objective.patient_contrast} {objective.research_objective}"
    )
    validate_no_inferred_current_exposure(
        combined,
        sample,
        context=f"Planner objective {objective.objective_id}",
    )
    validate_grounded_ddi_claim(
        objective.research_objective,
        sample,
        context=f"Planner objective {objective.objective_id}",
    )
    if (
        objective.research_objective.rstrip().endswith("?")
        or re.match(
            r"^(?:what|when|where|which|who|why|how|does|do|is|are|"
            r"can|could|should|would)\b",
            objective.research_objective,
            flags=re.IGNORECASE,
        )
    ):
        raise ValueError(
            f"Planner objective {objective.objective_id} generated a "
            "retrieval question instead of a declarative research objective."
        )
    allowed_medications = {
        name.casefold()
        for name in objective_allowed_medication_names(
            sample,
            objective.affected_candidates,
        )
    }
    foreign_medications = sorted(
        name
        for name in context_medication_names(sample)
        if (
            name.casefold() not in allowed_medications
            and mentions_name(combined, name)
        )
    )
    if foreign_medications:
        raise ValueError(
            f"Planner objective {objective.objective_id} introduced "
            "medications outside its affected candidates and grounded DDI "
            "partners: "
            + ", ".join(foreign_medications)
            + "."
        )
    combined_casefold = combined.casefold()
    irrelevant = [
        name
        for name in objective.affected_candidates
        if name.casefold() not in combined_casefold
    ]
    if irrelevant:
        raise ValueError(
            f"Planner objective {objective.objective_id} listed affected "
            "candidates that it did not explicitly address: "
            + ", ".join(irrelevant)
            + "."
        )


def validate_question(
    question: str,
    objective: ResearchObjective,
    sample: ResearchSample,
) -> None:
    normalized = normalized_text(question)
    affected = {
        name.casefold() for name in objective.affected_candidates
    }
    allowed_context = set(affected)
    for pair in sample.ddi_constraints:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        first = str(pair[0]).strip()
        second = str(pair[1]).strip()
        if first.casefold() in affected:
            allowed_context.add(second.casefold())
        if second.casefold() in affected:
            allowed_context.add(first.casefold())
    foreign_medications = sorted(
        name
        for name in context_medication_names(sample)
        if (
            name.casefold() not in allowed_context
            and mentions_name(normalized, name)
        )
    )
    if foreign_medications:
        raise ValueError(
            "Questioner introduced medication context outside the selected "
            "objective's affected candidates: "
            + ", ".join(foreign_medications)
            + "."
        )
    if re.search(
        r"\b(?:other|alternative)\s+"
        r"(?:agents?|drugs?|medications?|therap(?:y|ies))\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            "Questioner introduced unspecified alternative medication "
            "context outside the selected objective."
        )
    candidate_decision_pattern = any(
        re.search(
            rf"\b(?:need|necessity)\s+(?:for|of)\s+"
            rf"{re.escape(name)}\b",
            normalized,
            flags=re.IGNORECASE,
        )
        for name in objective.affected_candidates
    )
    if candidate_decision_pattern or any(
        pattern.search(normalized)
        for pattern in PRESCRIBING_DECISION_PATTERNS
    ):
        raise ValueError(
            "Questioner requested a prescribing decision instead of clinical "
            "or pharmacological evidence."
        )
    validate_no_inferred_current_exposure(
        normalized,
        sample,
        context="Questioner output",
    )
    lowered = normalized.casefold()
    monitoring_as_cause = (
        "monitoring" in lowered
        and any(
            term in lowered
            for term in (
                "affect",
                "influence",
                "impact",
                "provide evidence",
                "reassess",
            )
        )
        and any(
            term in lowered
            for term in (
                "appropriateness",
                "clinical outcome",
                "effectiveness",
                "safety",
                "use of",
            )
        )
    )
    if monitoring_as_cause:
        raise ValueError(
            "Questioner treated monitoring as a causal exposure rather than "
            "asking about the underlying organ function, laboratory values, "
            "dose, exposure, or disease severity."
        )
    question_casefold = normalized.casefold()
    if not any(
        name.casefold() in question_casefold
        for name in objective.affected_candidates
    ):
        raise ValueError(
            "Questioner output did not name an affected boundary medication "
            "from the selected research objective."
        )
