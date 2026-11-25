from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import dill

from meddr.constants import BOUNDARY_MAX, BOUNDARY_MIN, SIMILAR_PATIENTS
from meddr.schemas import ResearchSample


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return dill.load(handle)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def readable_names(vocabulary, names):
    resolved = []
    for index in range(len(vocabulary.idx2word)):
        code = str(vocabulary.idx2word[index])
        value = names.get(code)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Name mappings must cover every vocabulary code with readable text.")
        resolved.append(value.strip())
    return resolved


def clinical_visit(visit, names):
    return {key: [names[modality][index] for index in visit[modality]]
            for modality, key in enumerate(("diagnoses", "procedures", "medications"))}


class SimilarVisits:
    def __init__(self, records, names):
        self.visits, self.diagnoses, self.index = [], [], defaultdict(set)
        for patient_id, patient in enumerate(records):
            for visit in patient:
                readable = clinical_visit(visit, names)
                readable["patient_key"] = f"train:{patient_id}"
                diagnoses = {value.casefold() for value in readable["diagnoses"]}
                for diagnosis in diagnoses:
                    self.index[diagnosis].add(len(self.visits))
                self.visits.append(readable)
                self.diagnoses.append(diagnoses)

    def select(self, current, patient_key):
        diagnoses = {value.casefold() for value in current["diagnoses"]}
        candidates = set().union(*(self.index.get(code, set()) for code in diagnoses))
        best = {}
        for index in sorted(candidates):
            visit = self.visits[index]
            key = visit["patient_key"]
            if key == patient_key:
                continue
            other = self.diagnoses[index]
            score = len(diagnoses & other) / len(diagnoses | other)
            if key not in best or score > best[key][0]:
                best[key] = (score, index)
        ranked = sorted(best.values(), key=lambda item: (-item[0], item[1]))[:SIMILAR_PATIENTS]
        return [self.visits[index] for _, index in ranked]


def probability_rows(raw, count):
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError("Predictions must cover the complete medication vocabulary.")
    values = {}
    for identifier, probability in raw:
        if not isinstance(identifier, int) or not 0 <= identifier < count or identifier in values:
            raise ValueError("Medication indices in predictions must be unique and aligned.")
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Predicted probabilities must be finite and in [0, 1].")
        values[identifier] = probability
    return [values[index] for index in range(count)]


def prepare(args):
    vocabulary = load_pickle(args.data_dir / "voc_final.pkl")
    clinical_names = read_json(args.data_dir / "clinical_names.json")
    names = [readable_names(vocabulary[key], mapping) for key, mapping in (
        ("diag_voc", clinical_names["diagnoses"]),
        ("pro_voc", clinical_names["procedures"]),
        ("med_voc", read_json(args.medication_names)),
    )]
    if len(set(names[2])) != len(names[2]):
        raise ValueError("Medication names must be unique across ATC-3 classes.")
    training = load_pickle(args.data_dir / "records_train.pkl")
    target = training if args.split == "train" else load_pickle(args.data_dir / f"records_{args.split}.pkl")
    similar = SimilarVisits(training, names)
    predictions = read_json(args.predictions)
    if not isinstance(predictions, list) or len(predictions) != len(target):
        raise ValueError("Prediction patient count does not match the requested split.")
    indexed = {}
    for row in predictions:
        key = row["patient_id"]
        if not isinstance(key, int) or key in indexed or not 0 <= key < len(target):
            raise ValueError("Prediction patient indices must uniquely cover the split.")
        indexed[key] = row["visits"]
    written = skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for patient_id, patient in enumerate(target):
            predicted_visits = indexed[patient_id]
            by_visit = {row["visit_id"]: row for row in predicted_visits}
            if len(by_visit) != len(predicted_visits) or set(by_visit) != set(range(len(patient))):
                raise ValueError("Prediction visit indices must uniquely cover each patient sequence.")
            patient_key = f"{args.split}:{patient_id}"
            for visit_id, visit in enumerate(patient):
                probabilities = probability_rows(by_visit[visit_id]["predicted"], len(names[2]))
                if not any(BOUNDARY_MIN <= value <= BOUNDARY_MAX for value in probabilities):
                    skipped += 1
                    continue
                current = clinical_visit(visit, names)
                medications = [{"id": index, "name": names[2][index], "probability": value}
                               for index, value in enumerate(probabilities)]
                if args.split == "train":
                    for row in medications:
                        row["label"] = int(row["id"] in visit[2])
                sample = {
                    "sample_id": f"{patient_key}:visit:{visit_id}",
                    "patient": {
                        "patient_key": patient_key,
                        "current_visit": {key: current[key] for key in ("diagnoses", "procedures")},
                        "history": [clinical_visit(previous, names) for previous in patient[:visit_id]],
                    },
                    "base_model_outputs": {"medications": medications},
                    "similar_visit_candidates": similar.select(current, patient_key),
                    "ddi_constraints": [],
                }
                parsed = ResearchSample.from_dict(sample)
                if args.split == "train":
                    parsed.require_training_labels()
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                written += 1
    print(json.dumps({"research_visits": written, "baseline_only_visits": skipped}))


def main():
    parser = argparse.ArgumentParser(description="Convert GiantMed-style probabilities into MedDR visit inputs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--medication-names", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
