from __future__ import annotations

from collections import defaultdict

import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F


def molecular_graphs(smiles, medication_codes, radius=2):
    atom_ids, bond_ids, fingerprint_ids, edge_ids = ({}, {}, {}, {})

    def encode(table, value):
        return table.setdefault(value, len(table))

    fingerprints, sources, targets, molecule_index, drug_index = [], [], [], [], []
    for drug, code in enumerate(medication_codes):
        strings = smiles[code]
        strings = [strings] if isinstance(strings, str) else strings
        count = 0
        for value in sorted(set(strings)):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None or molecule.GetNumAtoms() == 0:
                continue
            molecule = Chem.AddHs(molecule)
            atoms = [encode(atom_ids, (atom.GetSymbol(), atom.GetIsAromatic()))
                     for atom in molecule.GetAtoms()]
            neighbors = defaultdict(list)
            for bond in molecule.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge = encode(bond_ids, str(bond.GetBondType()))
                neighbors[a].append((b, edge))
                neighbors[b].append((a, edge))
            nodes = atoms
            if len(atoms) == 1 or radius == 0:
                nodes = [encode(fingerprint_ids, (node, ())) for node in nodes]
            else:
                for _ in range(radius):
                    updated = [
                        encode(fingerprint_ids, (nodes[i], tuple(sorted(
                            (nodes[j], edge) for j, edge in neighbors[i]
                        )))) for i in range(len(atoms))
                    ]
                    neighbors = {
                        i: [(j, encode(edge_ids, (tuple(sorted((nodes[i], nodes[j]))), edge)))
                            for j, edge in neighbors[i]] for i in range(len(atoms))
                    }
                    nodes = updated
            offset = len(fingerprints)
            for bond in molecule.GetBonds():
                a, b = offset + bond.GetBeginAtomIdx(), offset + bond.GetEndAtomIdx()
                sources.extend((a, b))
                targets.extend((b, a))
            molecule_index.extend([len(drug_index)] * len(nodes))
            fingerprints.extend(nodes)
            drug_index.append(drug)
            count += 1
        if count == 0:
            raise ValueError("Every medication needs at least one valid molecular structure.")
    return {
        "fingerprints": torch.tensor(fingerprints, dtype=torch.long),
        "sources": torch.tensor(sources, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "molecule_index": torch.tensor(molecule_index, dtype=torch.long),
        "drug_index": torch.tensor(drug_index, dtype=torch.long),
        "fingerprint_count": len(fingerprint_ids),
    }


class MolecularEncoder(nn.Module):
    def __init__(self, graph, medication_count, dimension):
        super().__init__()
        for name in ("fingerprints", "sources", "targets", "molecule_index", "drug_index"):
            self.register_buffer(name, graph[name])
        self.register_buffer("counts", torch.bincount(
            graph["drug_index"], minlength=medication_count
        ).float().unsqueeze(1))
        self.embedding = nn.Embedding(graph["fingerprint_count"], dimension)
        self.layers = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(2)])

    def forward(self):
        atoms = self.embedding(self.fingerprints)
        for layer in self.layers:
            hidden = F.relu(layer(atoms))
            atoms = hidden + torch.zeros_like(hidden).index_add(0, self.targets, hidden[self.sources])
        molecules = atoms.new_zeros((self.drug_index.numel(), atoms.shape[1]))
        molecules = molecules.index_add(0, self.molecule_index, atoms)
        medications = atoms.new_zeros((self.counts.shape[0], atoms.shape[1]))
        return medications.index_add(0, self.drug_index, molecules) / self.counts


class SafeDrug(nn.Module):
    """Dual molecular representations with longitudinal patient encoding."""

    def __init__(self, vocabulary_sizes, ddi, mask, graph, dimension=64):
        super().__init__()
        medication_count = vocabulary_sizes[2]
        self.register_buffer("ddi", torch.as_tensor(ddi, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        if self.ddi.shape != (medication_count, medication_count):
            raise ValueError("DDI dimensions must match the medication vocabulary.")
        if self.mask.ndim != 2 or self.mask.shape[0] != medication_count or self.mask.shape[1] == 0:
            raise ValueError("The substructure mask must align with the medication vocabulary.")
        self.embeddings = nn.ModuleList([nn.Embedding(size, dimension) for size in vocabulary_sizes[:2]])
        self.dropout = nn.Dropout(0.5)
        self.encoders = nn.ModuleList([nn.GRU(dimension, dimension, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * dimension, dimension))
        self.molecular = MolecularEncoder(graph, medication_count, dimension)
        self.global_output = nn.Linear(medication_count, medication_count)
        self.global_norm = nn.LayerNorm(medication_count)
        self.local_transform = nn.Linear(dimension, self.mask.shape[1])
        self.local_output = nn.Linear(self.mask.shape[1], medication_count, bias=False)
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.local_output.weight, -medication_count ** -0.5, medication_count ** -0.5)

    def forward(self, visits, molecular_vectors=None):
        if not visits:
            raise ValueError("At least one visit is required.")
        representations = []
        for modality in range(2):
            sequence = torch.stack([
                self.dropout(self.embeddings[modality](torch.as_tensor(
                    visit[modality], dtype=torch.long, device=self.ddi.device
                ))).sum(dim=0) for visit in visits
            ]).unsqueeze(0)
            output, _ = self.encoders[modality](sequence)
            representations.append(output[:, -1])
        query = self.query(torch.cat(representations, dim=-1))
        molecular_vectors = self.molecular() if molecular_vectors is None else molecular_vectors
        match = torch.sigmoid(query @ molecular_vectors.T)
        global_scores = self.global_norm(match + self.global_output(match))
        local_scores = F.linear(torch.sigmoid(self.local_transform(query)), self.local_output.weight * self.mask)
        logits = local_scores * global_scores
        probabilities = torch.sigmoid(logits)
        ddi_loss = 0.0005 * (probabilities.T @ probabilities * self.ddi).sum()
        return logits, ddi_loss
