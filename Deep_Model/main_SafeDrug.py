from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import dill
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from Deep_Model.model import SafeDrug
from Deep_Model.model.safedrug import molecular_graphs


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return dill.load(handle)


def load_artifacts(data_dir):
    vocabulary = load_pickle(data_dir / "voc_final.pkl")
    codes = [[voc.idx2word[i] for i in range(len(voc.idx2word))]
             for voc in (vocabulary[key] for key in ("diag_voc", "pro_voc", "med_voc"))]
    ddi = np.asarray(load_pickle(data_dir / "ddi_A_final.pkl"), dtype=np.float32)
    mask = np.asarray(load_pickle(data_dir / "ddi_mask_H.pkl"), dtype=np.float32)
    smiles = load_pickle(data_dir / "atc3toSMILES.pkl")
    structures = {code: sorted(set([smiles[code]] if isinstance(smiles[code], str) else smiles[code]))
                  for code in codes[2]}
    digest = hashlib.sha256(json.dumps([codes, structures], sort_keys=True).encode())
    for matrix in (ddi, mask):
        if not np.isfinite(matrix).all() or not np.isin(matrix, (0, 1)).all():
            raise ValueError("Graph matrices must contain finite binary entries.")
        digest.update(str(matrix.shape).encode())
        digest.update(matrix.tobytes())
    if not np.array_equal(ddi, ddi.T) or np.any(np.diag(ddi)):
        raise ValueError("The DDI matrix must be symmetric with a zero diagonal.")
    return codes, ddi, mask, molecular_graphs(structures, codes[2]), digest.hexdigest()


def validate_records(records, sizes):
    if not isinstance(records, list) or not records or any(not patient for patient in records):
        raise ValueError("Records must contain non-empty patient sequences.")
    for patient in records:
        for visit in patient:
            if len(visit) != 3:
                raise ValueError("Each visit requires diagnoses, procedures, and medications.")
            for codes, size in zip(visit, sizes):
                if len(codes) != len(set(codes)) or any(
                    not isinstance(code, (int, np.integer)) or not 0 <= code < size for code in codes
                ):
                    raise ValueError("Visit codes must be unique vocabulary indices.")


def pair_counts(selected, adjacency):
    pairs = len(selected) * (len(selected) - 1) // 2
    interactions = sum(adjacency[a, b] > 0 for i, a in enumerate(selected) for b in selected[i + 1:])
    return int(interactions), pairs


def training_loss(model, sequence, target_ddi, kp, molecular_vectors):
    logits, ddi_loss = model(sequence, molecular_vectors)
    labels = sorted(sequence[-1][2])
    target = torch.zeros_like(logits)
    target[0, labels] = 1
    margin_target = torch.full_like(logits, -1, dtype=torch.long)
    margin_target[0, :len(labels)] = torch.tensor(labels, device=logits.device, dtype=torch.long)
    prediction_loss = 0.95 * F.binary_cross_entropy_with_logits(logits, target)
    prediction_loss = prediction_loss + 0.05 * F.multilabel_margin_loss(torch.sigmoid(logits), margin_target)
    selected = (torch.sigmoid(logits.detach())[0] >= 0.5).nonzero().flatten()
    subgraph = model.ddi[selected[:, None], selected[None, :]]
    pairs = selected.numel() * (selected.numel() - 1)
    rate = float(subgraph.sum()) / pairs if pairs else 0.0
    beta = max(0.0, min(1.0, 1.0 + (target_ddi - rate) / kp))
    return beta * prediction_loss + (1.0 - beta) * ddi_loss


@torch.no_grad()
def evaluate(model, records, adjacency, export=False):
    model.eval()
    molecular_vectors = model.molecular()
    patient_metrics, predictions = [], []
    total_interactions = total_pairs = 0
    for patient_id, patient in enumerate(records):
        metrics, visits = [], []
        for visit_id, visit in enumerate(patient):
            logits, _ = model(patient[:visit_id + 1], molecular_vectors)
            probability = torch.sigmoid(logits)[0].cpu().numpy()
            selected = np.flatnonzero(probability >= 0.5).tolist()
            actual = set(visit[2])
            overlap = len(actual.intersection(selected))
            union = len(actual.union(selected))
            labels = np.zeros_like(probability)
            labels[list(actual)] = 1
            metrics.append([
                overlap / union if union else 0.0,
                average_precision_score(labels, probability) if actual else 0.0,
                2 * overlap / (len(actual) + len(selected)) if actual or selected else 0.0,
                len(selected),
            ])
            interactions, pairs = pair_counts(selected, adjacency)
            total_interactions += interactions
            total_pairs += pairs
            if export:
                visits.append({"visit_id": visit_id, "predicted": [
                    [index, float(value)] for index, value in enumerate(probability)
                ]})
        patient_metrics.append(np.mean(metrics, axis=0))
        if export:
            predictions.append({"patient_id": patient_id, "visits": visits})
    result = dict(zip(("jaccard", "prauc", "f1", "medications"), np.mean(patient_metrics, axis=0).tolist()))
    result["ddi_rate"] = total_interactions / total_pairs if total_pairs else 0.0
    return result, predictions


def train(args, model, sizes, signature, adjacency):
    records = load_pickle(args.data_dir / "records_train.pkl")
    validation = load_pickle(args.data_dir / "records_valid.pkl")
    validate_records(records, sizes)
    validate_records(validation, sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    examples = [(patient, visit) for patient in range(len(records)) for visit in range(len(records[patient]))]
    best_jaccard = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        random.shuffle(examples)
        model.train()
        losses = []
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            molecules = model.molecular()
            loss = torch.stack([
                training_loss(model, records[patient][:visit + 1], args.target_ddi, args.kp, molecules)
                for patient, visit in batch
            ]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("SafeDrug produced a non-finite training loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, _ = evaluate(model, validation, adjacency)
        print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics}))
        if metrics["jaccard"] > best_jaccard:
            best_jaccard = metrics["jaccard"]
            torch.save({
                "state_dict": model.state_dict(), "dimension": args.dimension,
                "artifact_signature": signature, "epoch": epoch + 1,
            }, args.checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train SafeDrug or export frozen medication probabilities.")
    parser.add_argument("--mode", choices=("train", "predict"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--target-ddi", type=float, default=0.06)
    parser.add_argument("--kp", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.dimension, args.epochs, args.batch_size, args.learning_rate, args.kp) <= 0:
        parser.error("Training dimensions, counts, learning rate, and kp must be positive.")
    if not 0 <= args.target_ddi <= 1:
        parser.error("--target-ddi must be in [0, 1].")
    if args.mode == "predict" and args.output is None:
        parser.error("Prediction requires --output.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    codes, ddi, mask, graph, signature = load_artifacts(args.data_dir)
    sizes = [len(items) for items in codes]
    checkpoint = None
    if args.mode == "predict":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["artifact_signature"] != signature:
            raise ValueError("Checkpoint and preprocessing artifacts do not match.")
        args.dimension = checkpoint["dimension"]
    model = SafeDrug(sizes, ddi, mask, graph, args.dimension).to(args.device)
    if args.mode == "train":
        train(args, model, sizes, signature, ddi)
    else:
        model.load_state_dict(checkpoint["state_dict"])
        records = load_pickle(args.data_dir / f"records_{args.split}.pkl")
        validate_records(records, sizes)
        metrics, predictions = evaluate(model, records, ddi, export=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=True)
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
