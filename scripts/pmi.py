"""Índice de interação peptídeo–membrana (PMI) do pipeline PepMem-AI.

Entrada: descritores do peptídeo (carga, hidrofobicidade, momento) e da
membrana (carga superficial, colesterol, …). Saída: escalar PMI e seletividade.

Papel no pipeline: feature interpretável usada em ``build_pairs.py`` e nos
modelos baseline/multimodal.
"""

from __future__ import annotations

import math

# --- pesos empíricos do PMI (baseline interpretável) ---
DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 0.3, "delta": 0.4}


def compute_pmi(
    q_peptide: float,
    q_membrane: float,
    h_peptide: float,
    h_membrane: float,
    mu_h_peptide: float,
    cholesterol_membrane: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Combina carga, hidrofobicidade, momento e colesterol em índice PMI."""
    w = weights or DEFAULT_WEIGHTS
    return (
        w["alpha"] * q_peptide * abs(q_membrane)
        + w["beta"] * h_peptide * h_membrane
        + w["gamma"] * mu_h_peptide
        - w["delta"] * cholesterol_membrane
    )


def compute_pmi_sel(pmi_target: float, pmi_normal: float) -> float:
    """Seletividade: PMI no alvo menos PMI em célula mamífera normal."""
    return pmi_target - pmi_normal


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Momento hidrofóbico para hélice α (método de Eisenberg, ângulo 100°)."""
    from peptide_utils import AA_HYDRO

    if not seq:
        return 0.0
    angle = math.radians(angle_deg)
    hx = hy = 0.0
    for i, aa in enumerate(seq):
        h = AA_HYDRO.get(aa, 0.0)
        hx += h * math.cos(i * angle)
        hy += h * math.sin(i * angle)
    return math.sqrt(hx * hx + hy * hy) / len(seq)


def peptide_q(peptide_row) -> float:
    """Carga líquida do peptídeo (coluna anotada ou computada)."""
    if peptide_row.get("net_charge") is not None and not (isinstance(peptide_row.get("net_charge"), float) and math.isnan(peptide_row.get("net_charge"))):
        return float(peptide_row["net_charge"])
    val = peptide_row.get("net_charge_computed")
    return float(val) if val is not None else 0.0


def peptide_h(peptide_row) -> float:
    """Hidrofobicidade média do peptídeo (anotada ou computada)."""
    for key in ("hydrophobicity", "hydrophobicity_computed"):
        if key in peptide_row and peptide_row[key] is not None:
            try:
                return float(peptide_row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def peptide_mu_h(peptide_row) -> float:
    """Momento hidrofóbico do peptídeo (anotado ou calculado da sequência)."""
    if peptide_row.get("hydrophobic_moment") is not None:
        try:
            return float(peptide_row["hydrophobic_moment"])
        except (TypeError, ValueError):
            pass
    seq = peptide_row.get("sequence")
    if seq:
        return hydrophobic_moment(str(seq))
    return 0.0
