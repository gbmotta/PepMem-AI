"""Load and validate bench (lab) MIC/MBC measurements."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "bench"

MIC_REQUIRED = {"target_id", "endpoint", "value"}


def _bench_path(name: str) -> Path:
    return BENCH_DIR / name


def load_csv(name: str) -> pd.DataFrame:
    path = _bench_path(name)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#")
    df = df.dropna(how="all")
    return df


def normalize_sequence(seq: str | None) -> str | None:
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return None
    from peptide_utils import normalize_sequence as norm

    return norm(str(seq))


def load_bench_peptides() -> pd.DataFrame:
    return load_csv("peptides_bench.csv")


def load_bench_targets() -> pd.DataFrame:
    return load_csv("targets_bench.csv")


def load_bench_mic() -> pd.DataFrame:
    df = load_csv("mic_bench.csv")
    if df.empty:
        return df
    return validate_mic_df(df)


def validate_mic_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    missing = MIC_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"mic_bench.csv: colunas obrigatórias ausentes: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        peptide_id = row.get("peptide_id")
        if pd.isna(peptide_id) or not str(peptide_id).strip():
            peptide_id = None
        else:
            peptide_id = str(peptide_id).strip()

        sequence = normalize_sequence(row.get("sequence"))
        if not peptide_id and not sequence:
            raise ValueError(f"Linha {i + 2}: informe peptide_id ou sequence")

        endpoint = str(row["endpoint"]).strip().upper()
        allowed = {"MIC", "MBC", "CC50", "IC50", "HEMOLYSIS", "BIOFILM_INHIB"}
        if endpoint not in allowed:
            raise ValueError(
                f"Linha {i + 2}: endpoint inválido '{endpoint}' "
                f"(use {', '.join(sorted(allowed))})"
            )

        try:
            value = float(row["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Linha {i + 2}: value numérico inválido") from exc

        # Unidades padrão por endpoint
        default_unit = {
            "MIC": "uM",
            "MBC": "uM",
            "CC50": "uM",
            "IC50": "uM",
            "HEMOLYSIS": "percent",
            "BIOFILM_INHIB": "percent",
        }.get(endpoint, "uM")
        unit = default_unit if pd.isna(row.get("unit")) else str(row.get("unit")).strip()

        target_id = str(row["target_id"]).strip()
        rows.append(
            {
                "peptide_id": peptide_id,
                "sequence": sequence,
                "name": None if pd.isna(row.get("name")) else str(row.get("name")).strip(),
                "net_charge": None if pd.isna(row.get("net_charge")) else float(row.get("net_charge")),
                "target_id": target_id,
                "target": None if pd.isna(row.get("target")) else str(row.get("target")).strip(),
                "target_type": None if pd.isna(row.get("target_type")) else str(row.get("target_type")).strip(),
                "endpoint": endpoint,
                "value": value,
                "unit": unit,
                "assay": "microdilution" if pd.isna(row.get("assay")) else str(row.get("assay")).strip(),
                "reference": "bancada" if pd.isna(row.get("reference")) else str(row.get("reference")).strip(),
                "date": None if pd.isna(row.get("date")) else str(row.get("date")).strip(),
                "notes": None if pd.isna(row.get("notes")) else str(row.get("notes")).strip(),
            }
        )

    return pd.DataFrame(rows)


def next_peptide_id(existing_ids: set[str]) -> str:
    nums = []
    for pid in existing_ids:
        m = re.match(r"^P(\d+)$", str(pid))
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    return f"P{n:02d}"


def resolve_peptides(
    mic_df: pd.DataFrame,
    project_df: pd.DataFrame,
    bench_peptides_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return updated project_df, resolved mic_df with peptide_id, list of new ids."""
    project = project_df.copy()
    id_to_seq = project.set_index("peptide_id")["sequence"].to_dict()
    seq_to_id = {v.upper(): k for k, v in id_to_seq.items() if v}

    for _, row in bench_peptides_df.iterrows():
        pid = str(row["peptide_id"]).strip()
        seq = normalize_sequence(row.get("sequence"))
        if not seq:
            raise ValueError(f"peptides_bench.csv: {pid} sem sequence")
        if pid in id_to_seq and id_to_seq[pid].upper() != seq:
            raise ValueError(f"peptides_bench.csv: {pid} sequence conflita com projeto")
        if pid not in id_to_seq:
            project = pd.concat([project, pd.DataFrame([row.to_dict()])], ignore_index=True)
            id_to_seq[pid] = seq
            seq_to_id[seq.upper()] = pid

    if mic_df.empty and bench_peptides_df.empty:
        return project, mic_df, []

    resolved = mic_df.copy()
    new_ids: list[str] = []
    existing = set(project["peptide_id"].astype(str))

    for idx, row in resolved.iterrows():
        pid = row.get("peptide_id")
        seq = row.get("sequence")
        if pid and str(pid) in id_to_seq:
            resolved.at[idx, "peptide_id"] = str(pid)
            if not seq or (isinstance(seq, float) and pd.isna(seq)):
                resolved.at[idx, "sequence"] = id_to_seq[str(pid)]
            continue
        if seq and not (isinstance(seq, float) and pd.isna(seq)):
            seq_u = str(seq).upper()
            if seq_u in seq_to_id:
                resolved.at[idx, "peptide_id"] = seq_to_id[seq_u]
                resolved.at[idx, "sequence"] = seq_u
                continue
            new_id = next_peptide_id(existing)
            existing.add(new_id)
            new_ids.append(new_id)
            name = row.get("name") or f"bench_{new_id}"
            net_charge = row.get("net_charge")
            new_row = {
                "peptide_id": new_id,
                "name": name,
                "parent": "bench",
                "sequence_raw": seq_u,
                "sequence": seq_u,
                "net_charge": net_charge,
                "c_terminal": "NH2",
                "source": "bancada",
                "dataset": "PepMem-Base-Project",
            }
            project = pd.concat([project, pd.DataFrame([new_row])], ignore_index=True)
            id_to_seq[new_id] = seq_u
            seq_to_id[seq_u] = new_id
            resolved.at[idx, "peptide_id"] = new_id
            resolved.at[idx, "sequence"] = seq_u

    return project, resolved, new_ids


def bench_endpoints_records(mic_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in mic_df.iterrows():
        records.append(
            {
                "peptide_id": row["peptide_id"],
                "sequence": row["sequence"],
                "target_id": row["target_id"],
                "target": row.get("target"),
                "target_type": row.get("target_type"),
                "endpoint": row["endpoint"],
                "value": row["value"],
                "unit": row.get("unit", "uM"),
                "assay": row.get("assay", "microdilution"),
                "reference": row.get("reference", "bancada"),
                "source": "bancada",
                "confidence": "experimental",
                "split": "bench",
                "date": row.get("date"),
                "notes": row.get("notes"),
            }
        )
    return records

