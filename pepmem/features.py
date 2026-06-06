"""Feature engineering for PepMem-AI inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CLASSIC_FEATURES = [
    "q_peptide",
    "h_peptide",
    "mu_h_peptide",
    "surface_charge",
    "anionic_fraction",
    "cholesterol",
    "lps",
    "peptidoglycan",
    "ergosterol",
    "viral_envelope",
    "pmi",
]


def load_targets() -> pd.DataFrame:
    project = pd.read_parquet(ROOT / "data" / "processed" / "project_membrane_targets.parquet")
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    extra_ids = pairs[~pairs["target_id"].isin(project["target_id"])]["target_id"].unique()
    if len(extra_ids):
        # Descritores das cepas MDR já presentes nos pares
        cols = [
            "target_id", "target", "target_type", "surface_charge", "anionic_fraction",
            "cholesterol", "lps", "peptidoglycan", "ergosterol", "viral_envelope",
        ]
        extra = pairs.drop_duplicates("target_id")[cols]
        project = pd.concat([project, extra], ignore_index=True).drop_duplicates("target_id")
    return project


def peptide_row_from_sequence(sequence: str, net_charge: float | None = None) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from peptide_utils import add_descriptor_columns, normalize_sequence
    from pmi import hydrophobic_moment, peptide_q

    seq = normalize_sequence(sequence)
    if not seq:
        raise ValueError("Sequência inválida ou vazia.")

    base = {
        "sequence": seq,
        "net_charge": net_charge,
        "hydrophobicity_computed": None,
        "hydrophobic_moment": hydrophobic_moment(seq),
    }
    df = add_descriptor_columns(pd.DataFrame([base]))
    row = df.iloc[0].to_dict()
    if net_charge is None:
        row["net_charge"] = row.get("net_charge_computed")
    row["q_peptide"] = peptide_q(row)
    row["h_peptide"] = float(row.get("hydrophobicity_computed") or 0)
    row["mu_h_peptide"] = float(row.get("hydrophobic_moment") or 0)
    return row


def pair_features(peptide: dict[str, Any], target: pd.Series) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from pmi import compute_pmi

    h_m = 0.5
    chol = float(target.get("cholesterol") or 0)
    pmi = compute_pmi(
        peptide["q_peptide"],
        float(target["surface_charge"]),
        peptide["h_peptide"],
        h_m,
        peptide["mu_h_peptide"],
        chol,
    )
    return {
        "peptide_id": peptide.get("peptide_id"),
        "sequence": peptide["sequence"],
        "target_id": target["target_id"],
        "target": target["target"],
        "target_type": target["target_type"],
        "q_peptide": peptide["q_peptide"],
        "h_peptide": peptide["h_peptide"],
        "mu_h_peptide": peptide["mu_h_peptide"],
        "surface_charge": float(target["surface_charge"]),
        "anionic_fraction": float(target.get("anionic_fraction") or 0),
        "cholesterol": chol,
        "lps": float(target.get("lps") or 0),
        "peptidoglycan": float(target.get("peptidoglycan") or 0),
        "ergosterol": float(target.get("ergosterol") or 0),
        "viral_envelope": float(target.get("viral_envelope") or 0),
        "pmi": pmi,
    }


def vectorize(features: dict[str, Any], embedding: np.ndarray | None, use_embeddings: bool) -> np.ndarray:
    classic = np.array([features[k] for k in CLASSIC_FEATURES], dtype=np.float32)
    if use_embeddings and embedding is not None:
        return np.concatenate([classic, embedding.astype(np.float32)])
    return classic
