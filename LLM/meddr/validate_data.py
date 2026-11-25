from __future__ import annotations

import argparse
import json
import os

from meddr.io import load_pickle, load_samples
from meddr.schemas import validate_and_align_ddi_constraints


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description="Validate MedDR visit records and DDI alignment."
    )
    data_inputs = arguments.add_argument_group("Data inputs")
    data_inputs.add_argument("--data", required=True)
    data_inputs.add_argument(
        "--ddi-matrix",
        default=os.getenv("MEDDR_DDI_MATRIX", "data/ready/ddi_A_final.pkl"),
        help="Preprocessed DDI adjacency file aligned with the medication vocabulary.",
    )
    arguments.add_argument(
        "--mode",
        choices=("training", "inference"),
        default="training",
    )
    return arguments


def main() -> None:
    args = parser().parse_args()
    samples = load_samples(args.data)
    if not samples:
        raise ValueError("No samples were loaded.")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique.")
    if args.mode == "training":
        for sample in samples:
            sample.require_training_labels()
    ddi_matrix = load_pickle(args.ddi_matrix)
    validate_and_align_ddi_constraints(samples, ddi_matrix)
    summary = {
        "mode": args.mode,
        "sample_count": len(samples),
        "boundary_medication_count": sum(
            len(sample.boundary_medications) for sample in samples
        ),
        "history_visit_count": sum(
            len(sample.history) for sample in samples
        ),
        "selected_similar_visit_count": sum(
            len(sample.similar_visits) for sample in samples
        ),
        "ddi_constraint_count": sum(
            len(sample.ddi_constraints) for sample in samples
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
