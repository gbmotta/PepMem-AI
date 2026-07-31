"""Engenharia de features para inferência PepMem-AI.

Constrói o vetor clássico (carga, hidrofobicidade, momento, descritores de
membrana e PMI) a partir de uma sequência e de um ``target_id``. Opcionalmente
concatena embeddings ESM-2 via ``vectorize``.

Papel no pipeline: compartilhado pelo ``PepMemPredictor``, pelo treino
multimodal e pelas explicações SHAP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# --- features tabulares do modelo baseline (mesma ordem no treino) ---
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
    """Carrega alvos de membrana do projeto e inclui cepas extras presentes nos pares."""
    project = pd.read_parquet(ROOT / "data" / "processed" / "project_membrane_targets.parquet")
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    extra_ids = pairs[~pairs["target_id"].isin(project["target_id"])]["target_id"].unique()
    if len(extra_ids):
        # Descritores das cepas MDR / bancada já normalizados nos pares
        cols = [
            "target_id", "target", "target_type", "surface_charge", "anionic_fraction",
            "cholesterol", "lps", "peptidoglycan", "ergosterol", "viral_envelope",
        ]
        extra = pairs.drop_duplicates("target_id")[cols]
        project = pd.concat([project, extra], ignore_index=True).drop_duplicates("target_id")
    return project


def peptide_row_from_sequence(sequence: str, net_charge: float | None = None) -> dict[str, Any]:
    """Normaliza a sequência e calcula q, h e μH (momento hidrofóbico).

    Se ``net_charge`` for informado, prevalece sobre a carga estimada.
    """
    from pepmem.peptide_utils import add_descriptor_columns, normalize_sequence
    from pepmem.pmi import hydrophobic_moment, peptide_q

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
    from pepmem.pmi import peptide_h, peptide_mu_h

    row["q_peptide"] = peptide_q(row)
    row["h_peptide"] = peptide_h(row)
    row["mu_h_peptide"] = peptide_mu_h(row)
    return row


def pair_features(peptide: dict[str, Any], target: pd.Series) -> dict[str, Any]:
    """Junta descritores do peptídeo e do alvo e calcula o PMI do par."""
    from pepmem.pmi import compute_pmi, membrane_h_proxy, sterol_penalty

    explicit_h = target.get("membrane_hydrophobicity")
    if explicit_h is None or (isinstance(explicit_h, float) and pd.isna(explicit_h)):
        explicit_h = target.get("hydrophobicity")
    h_m = membrane_h_proxy(target.get("target_type"), explicit=explicit_h)
    chol = float(target.get("cholesterol") or 0)
    ergo = float(target.get("ergosterol") or 0)
    sterol = sterol_penalty(chol, ergo)
    pmi = compute_pmi(
        peptide["q_peptide"],
        float(target["surface_charge"]),
        peptide["h_peptide"],
        h_m,
        peptide["mu_h_peptide"],
        sterol,
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
        "h_membrane": h_m,
        "surface_charge": float(target["surface_charge"]),
        "anionic_fraction": float(target.get("anionic_fraction") or 0),
        "cholesterol": chol,
        "lps": float(target.get("lps") or 0),
        "peptidoglycan": float(target.get("peptidoglycan") or 0),
        "ergosterol": ergo,
        "sterol_penalty": sterol,
        "viral_envelope": float(target.get("viral_envelope") or 0),
        "pmi": pmi,
    }


def vectorize(features: dict[str, Any], embedding: np.ndarray | None, use_embeddings: bool) -> np.ndarray:
    """Monta o vetor de entrada do RF: features clássicas (+ ESM-2 se ativo)."""
    classic = np.array([features[k] for k in CLASSIC_FEATURES], dtype=np.float32)
    if use_embeddings and embedding is not None:
        return np.concatenate([classic, embedding.astype(np.float32)])
    return classic
