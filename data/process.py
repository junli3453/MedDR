from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return dill.load(handle)


def dump_pickle(value, path):
    with Path(path).open("wb") as handle:
        dill.dump(value, handle)


def vocabulary(values):
    words = list(dict.fromkeys(word for row in values for word in row))
    return SimpleNamespace(
        idx2word=dict(enumerate(words)),
        word2idx={word: index for index, word in enumerate(words)},
    )


def clinical_codes(raw_dir, kind):
    filename = "DIAGNOSES_ICD.csv" if kind == "diagnoses" else "PROCEDURES_ICD.csv"
    dictionary = "D_ICD_DIAGNOSES.csv" if kind == "diagnoses" else "D_ICD_PROCEDURES.csv"
    frame = pd.read_csv(raw_dir / filename, dtype={"ICD9_CODE": str})
    frame = frame.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    frame = frame.drop_duplicates()
    if kind == "diagnoses":
        counts = frame.groupby("ICD9_CODE").size().sort_values(ascending=False)
        frame = frame[frame.ICD9_CODE.isin(counts.index[:2000])]
    names = pd.read_csv(raw_dir / dictionary, dtype={"ICD9_CODE": str})
    names = names.dropna(subset=["ICD9_CODE", "LONG_TITLE"])
    name_map = dict(zip(names.ICD9_CODE, names.LONG_TITLE))
    frame = frame[frame.ICD9_CODE.isin(name_map)]
    frame = frame.sort_values(["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    grouped = frame.groupby(["SUBJECT_ID", "HADM_ID"]).ICD9_CODE.agg(
        lambda values: list(dict.fromkeys(values))
    ).rename(kind).reset_index()
    return grouped, name_map


def medications(raw_dir):
    frame = pd.read_csv(raw_dir / "PRESCRIPTIONS.csv", dtype={"NDC": str})
    frame = frame[frame.NDC.notna() & (frame.NDC != "0")]
    required = ["SUBJECT_ID", "HADM_ID", "NDC", "DRUG", "STARTDATE"]
    frame = frame.dropna(subset=required).drop_duplicates()
    frame["STARTDATE"] = pd.to_datetime(frame.STARTDATE, errors="raise")
    frame = frame.sort_values(["SUBJECT_ID", "STARTDATE", "HADM_ID", "DRUG"])
    eligible = frame.groupby("SUBJECT_ID").HADM_ID.nunique()
    frame = frame[frame.SUBJECT_ID.isin(eligible[eligible >= 2].index)]
    with (raw_dir / "ndc2RXCUI.txt").open(encoding="utf-8") as handle:
        mapping = ast.literal_eval(handle.read())
    if not isinstance(mapping, dict):
        raise ValueError("The NDC mapping must be a dictionary.")
    mapping = {str(key): str(value) for key, value in mapping.items()}
    frame["RXCUI"] = frame.NDC.map(mapping)
    atc = pd.read_csv(raw_dir / "ndc2atc_level4.csv", dtype={"RXCUI": str})
    atc = atc.dropna(subset=["RXCUI", "ATC4"]).drop_duplicates("RXCUI")
    frame = frame.merge(atc[["RXCUI", "ATC4"]], on="RXCUI", how="inner")
    frame["ATC3"] = frame.ATC4.str[:4]
    raw_smiles = load_pickle(raw_dir / "idx2SMILES.pkl")
    smiles = {}
    for code, strings in raw_smiles.items():
        strings = [strings] if isinstance(strings, str) else strings
        valid = sorted({str(value) for value in strings if Chem.MolFromSmiles(str(value)) is not None})
        if valid:
            smiles[str(code)] = valid
    frame = frame[frame.ATC3.isin(smiles)]
    counts = frame.groupby("ATC3").size().sort_values(ascending=False)
    frame = frame[frame.ATC3.isin(counts.index[:300])]
    times = frame.groupby(["SUBJECT_ID", "HADM_ID"]).STARTDATE.min().rename("time")
    frame = frame.drop_duplicates(["SUBJECT_ID", "HADM_ID", "ATC3"])
    grouped = frame.groupby(["SUBJECT_ID", "HADM_ID"]).ATC3.agg(list).rename("medications")
    return grouped.to_frame().join(times).reset_index(), smiles


def ddi_matrix(raw_dir, med_vocabulary):
    cid_to_atc = defaultdict(set)
    with (raw_dir / "drug-atc.csv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                cid_to_atc[row[0]].update(code[:4] for code in row[1:])
    interactions = pd.read_csv(raw_dir / "drug-DDI.csv", dtype={"STITCH 1": str, "STITCH 2": str})
    counts = interactions.groupby(["Polypharmacy Side Effect", "Side Effect Name"]).size()
    selected = counts.sort_values(ascending=False).tail(40).reset_index()
    # Retain the same side-effect category selection as the reference pipeline.
    pairs = interactions.merge(selected[["Side Effect Name"]], on="Side Effect Name")
    pairs = pairs[["STITCH 1", "STITCH 2"]].drop_duplicates()
    adjacency = np.zeros((len(med_vocabulary.idx2word),) * 2, dtype=np.float32)
    for first, second in pairs.itertuples(index=False, name=None):
        for code_a in cid_to_atc[first]:
            for code_b in cid_to_atc[second]:
                a = med_vocabulary.word2idx.get(code_a)
                b = med_vocabulary.word2idx.get(code_b)
                if a is not None and b is not None and a != b:
                    adjacency[a, b] = adjacency[b, a] = 1
    return adjacency


def substructure_mask(smiles, med_vocabulary):
    fragments = []
    for index in range(len(med_vocabulary.idx2word)):
        fragments.append(set().union(*(
            BRICS.BRICSDecompose(Chem.MolFromSmiles(value))
            for value in smiles[med_vocabulary.idx2word[index]]
        )))
    fragment_names = sorted(set().union(*fragments))
    columns = {name: index for index, name in enumerate(fragment_names)}
    mask = np.zeros((len(fragments), len(columns)), dtype=np.float32)
    for row, names in enumerate(fragments):
        for name in names:
            mask[row, columns[name]] = 1
    return mask


def process(raw_dir, output_dir):
    med, smiles = medications(raw_dir)
    diag, diagnosis_names = clinical_codes(raw_dir, "diagnoses")
    proc, procedure_names = clinical_codes(raw_dir, "procedures")
    frame = med.merge(diag, on=["SUBJECT_ID", "HADM_ID"]).merge(proc, on=["SUBJECT_ID", "HADM_ID"])
    eligible = frame.groupby("SUBJECT_ID").HADM_ID.nunique()
    frame = frame[frame.SUBJECT_ID.isin(eligible[eligible >= 2].index)]
    frame = frame.sort_values(["SUBJECT_ID", "time", "HADM_ID"])
    if frame.empty:
        raise ValueError("No patients remain after preprocessing.")
    voc = {key: vocabulary(frame[column]) for key, column in (
        ("diag_voc", "diagnoses"), ("pro_voc", "procedures"), ("med_voc", "medications")
    )}
    records = []
    for _, patient in frame.groupby("SUBJECT_ID", sort=False):
        records.append([
            [[voc[key].word2idx[code] for code in row[column]] for key, column in (
                ("diag_voc", "diagnoses"), ("pro_voc", "procedures"), ("med_voc", "medications")
            )] for _, row in patient.iterrows()
        ])
    split = int(len(records) * 2 / 3)
    middle = split + (len(records) - split) // 2
    if not 0 < split < middle < len(records):
        raise ValueError("At least six patients are needed for non-empty splits.")
    ddi = ddi_matrix(raw_dir, voc["med_voc"])
    smiles = {code: smiles[code] for code in voc["med_voc"].word2idx}
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "records_final.pkl": records,
        "records_train.pkl": records[:split],
        "records_test.pkl": records[split:middle],
        "records_valid.pkl": records[middle:],
        "voc_final.pkl": voc,
        "atc3toSMILES.pkl": smiles,
        "ddi_A_final.pkl": ddi,
        "ddi_mask_H.pkl": substructure_mask(smiles, voc["med_voc"]),
    }
    for filename, value in artifacts.items():
        dump_pickle(value, output_dir / filename)
    names = {"diagnoses": diagnosis_names, "procedures": procedure_names}
    with (output_dir / "clinical_names.json").open("w", encoding="utf-8") as handle:
        json.dump(names, handle, ensure_ascii=True, indent=2)
    print(f"Prepared {len(records)} patients and {sum(map(len, records))} visits.")


def main():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-III records following the GiantMed workflow.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    process(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
