"""Índice de interação peptídeo–membrana (PMI) do pipeline PepMem-AI.

Entrada: descritores do peptídeo (carga, hidrofobicidade, momento) e da
membrana (carga superficial, colesterol, …). Saída: escalar PMI e seletividade.

Usado em ``build_pairs``, treino e inferência (``pepmem.features``).
"""

from __future__ import annotations

import math
from typing import Any

# --- pesos empíricos do PMI (baseline interpretável) ---
DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 0.3, "delta": 0.4}

# Proxy de hidrofobicidade da membrana por tipo de alvo (substitui h_m=0,5 fixo).
# Valores relativos tipificados — não são medidas experimentais.
MEMBRANE_H_BY_TYPE: dict[str, float] = {
    "Gram+": 0.45,
    "Gram-": 0.40,
    "fungo": 0.55,
    "parasita": 0.55,
    "mamífero normal": 0.65,
    "mamífero (hemólise)": 0.65,
    "célula tumoral": 0.55,
    "vírus envelopado": 0.60,
    "vírus/membrana": 0.60,
    "eucariota": 0.60,
    "organelle": 0.50,
    "arquea": 0.50,
    "outro": 0.50,
}


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    try:
        return bool(isinstance(val, float) and math.isnan(val))
    except (TypeError, ValueError):
        return False


def membrane_h_proxy(target_type: str | None, explicit: float | None = None) -> float:
    """Hidrofobicidade da membrana: coluna explícita ou proxy por ``target_type``."""
    if explicit is not None and not _is_missing(explicit):
        return float(explicit)
    if not target_type:
        return 0.5
    return MEMBRANE_H_BY_TYPE.get(str(target_type).strip(), 0.5)


def sterol_penalty(cholesterol: float = 0.0, ergosterol: float = 0.0) -> float:
    """Esterol efetivo no PMI: colesterol + fração de ergosterol (fungo/parasita)."""
    c = 0.0 if _is_missing(cholesterol) else float(cholesterol)
    e = 0.0 if _is_missing(ergosterol) else float(ergosterol)
    return c + 0.8 * e


def compute_pmi(
    q_peptide: float,
    q_membrane: float,
    h_peptide: float,
    h_membrane: float,
    mu_h_peptide: float,
    cholesterol_membrane: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Combina carga, hidrofobicidade, momento e esterol em índice PMI."""
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
    from pepmem.peptide_utils import AA_HYDRO

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
    """Carga líquida: anotada (`net_charge`) prevalece; senão estimada da sequência."""
    if not _is_missing(peptide_row.get("net_charge")):
        return float(peptide_row["net_charge"])
    val = peptide_row.get("net_charge_computed")
    if not _is_missing(val):
        return float(val)
    seq = peptide_row.get("sequence")
    if seq:
        from pepmem.peptide_utils import compute_net_charge

        est = compute_net_charge(str(seq))
        return float(est) if est is not None else 0.0
    return 0.0


def peptide_h(peptide_row) -> float:
    """Hidrofobicidade média Kyte–Doolittle (sempre da sequência se faltar computed).

    A coluna ``hydrophobicity`` do Quadro CNPq (~0–1) NÃO é a escala KD; por isso
    preferimos ``hydrophobicity_computed`` / cálculo da sequência.
    """
    from pepmem.peptide_utils import compute_mean_hydrophobicity

    for key in ("hydrophobicity_computed",):
        if key in peptide_row and not _is_missing(peptide_row.get(key)):
            try:
                return float(peptide_row[key])
            except (TypeError, ValueError):
                pass
    seq = peptide_row.get("sequence")
    if seq:
        h = compute_mean_hydrophobicity(str(seq))
        if h is not None:
            return float(h)
    # fallback: anotação só se parecer escala KD
    ann = peptide_row.get("hydrophobicity")
    if not _is_missing(ann):
        try:
            return float(ann)
        except (TypeError, ValueError):
            pass
    return 0.0


def peptide_mu_h(peptide_row) -> float:
    """Momento hidrofóbico: anotado se válido; senão Eisenberg na sequência."""
    ann = peptide_row.get("hydrophobic_moment")
    if not _is_missing(ann):
        try:
            return float(ann)
        except (TypeError, ValueError):
            pass
    seq = peptide_row.get("sequence")
    if seq:
        return hydrophobic_moment(str(seq))
    return 0.0
