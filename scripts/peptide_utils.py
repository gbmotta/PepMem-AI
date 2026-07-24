"""Utilitários compartilhados de parsing e descritores de peptídeos.

Entrada: sequências em texto ou FASTA. Saída: sequências normalizadas e
colunas de descritores (carga, hidrofobicidade, frações, comprimento).

Papel no pipeline: biblioteca usada por ``build_datasets.py``, ``bench_mic.py``
e ``pmi.py`` para normalização e engenharia de features clássicas.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- escalas de aminoácidos (Kyte-Doolittle e carga a pH ~7) ---
AA_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_CHARGE = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
CATIONIC = set("KRH")
HYDROPHOBIC = set("AVILMFWYP")
VALID_AA = set(AA_HYDRO)


def normalize_sequence(seq: str | None) -> str | None:
    """Remove caracteres inválidos e retorna sequência maiúscula ou None."""
    if not seq:
        return None
    seq = seq.strip().upper().replace(" ", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq or None


def compute_net_charge(seq: str | None, ph_assumption: float = 7.0) -> float | None:
    """Estima carga líquida somando contribuições de resíduos ionizáveis."""
    if not seq:
        return None
    charge = 0.0
    for aa in seq:
        if aa in AA_CHARGE:
            charge += AA_CHARGE[aa]
    return round(charge, 2)


def compute_length(seq: str | None) -> int | None:
    """Retorna comprimento da sequência ou None."""
    return len(seq) if seq else None


def compute_cationic_fraction(seq: str | None) -> float | None:
    """Fração de resíduos catiônicos (K, R, H) na sequência."""
    if not seq:
        return None
    return round(sum(aa in CATIONIC for aa in seq) / len(seq), 4)


def compute_hydrophobic_fraction(seq: str | None) -> float | None:
    """Fração de resíduos hidrofóbicos na sequência."""
    if not seq:
        return None
    return round(sum(aa in HYDROPHOBIC for aa in seq) / len(seq), 4)


def compute_mean_hydrophobicity(seq: str | None) -> float | None:
    """Hidrofobicidade média (escala Kyte-Doolittle) dos resíduos válidos."""
    if not seq:
        return None
    vals = [AA_HYDRO[aa] for aa in seq if aa in AA_HYDRO]
    return round(sum(vals) / len(vals), 4) if vals else None


def add_descriptor_columns(df, sequence_col: str = "sequence"):
    """Adiciona colunas de descritores computados a partir da sequência."""
    df = df.copy()
    df[sequence_col] = df[sequence_col].map(normalize_sequence)
    df["length"] = df[sequence_col].map(compute_length)
    df["net_charge_computed"] = df[sequence_col].map(compute_net_charge)
    df["cationic_fraction"] = df[sequence_col].map(compute_cationic_fraction)
    df["hydrophobic_fraction"] = df[sequence_col].map(compute_hydrophobic_fraction)
    df["hydrophobicity_computed"] = df[sequence_col].map(compute_mean_hydrophobicity)
    return df


def parse_fasta(path: Path) -> list[dict]:
    """Lê arquivo FASTA e retorna lista de registros com header, apd_id e sequence."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_fasta_text(text)


def parse_fasta_text(text: str) -> list[dict]:
    """Parseia conteúdo FASTA (string ou bytes decodificados) → lista de registros."""
    records: list[dict] = []
    header: str | None = None
    seq_parts: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_fasta_record(header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)

    if header is not None:
        records.append(_fasta_record(header, "".join(seq_parts)))

    return records


def _fasta_record(header: str, sequence: str) -> dict:
    """Monta um registro FASTA normalizado; detecta ID APD no cabeçalho."""
    sequence = normalize_sequence(sequence)
    apd_id = header if re.match(r"AP\d+", header) else None
    return {
        "header": header,
        "apd_id": apd_id,
        "sequence": sequence,
    }
