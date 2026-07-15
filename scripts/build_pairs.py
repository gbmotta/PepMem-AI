#!/usr/bin/env python3
"""Build peptide-membrane pairs with PMI features for modeling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pmi import compute_pmi, compute_pmi_sel, peptide_h, peptide_mu_h, peptide_q


from bench_mic import load_bench_targets


LITERATURE_TARGETS = [
    {"target_id": "S_aureus_UFPEDA1040", "target": "Staphylococcus aureus UFPEDA1040", "target_type": "Gram+", "surface_charge": -0.80, "anionic_fraction": 0.60, "lps": 0, "peptidoglycan": 1, "teichoic_acid": 1, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
    {"target_id": "S_aureus_UFPEDA1051", "target": "Staphylococcus aureus UFPEDA1051", "target_type": "Gram+", "surface_charge": -0.80, "anionic_fraction": 0.60, "lps": 0, "peptidoglycan": 1, "teichoic_acid": 1, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
    {"target_id": "S_aureus_UFPEDA1058", "target": "Staphylococcus aureus UFPEDA1058", "target_type": "Gram+", "surface_charge": -0.80, "anionic_fraction": 0.60, "lps": 0, "peptidoglycan": 1, "teichoic_acid": 1, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
    {"target_id": "S_aureus_UFPEDA1059", "target": "Staphylococcus aureus UFPEDA1059", "target_type": "Gram+", "surface_charge": -0.80, "anionic_fraction": 0.60, "lps": 0, "peptidoglycan": 1, "teichoic_acid": 1, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
    {"target_id": "P_aeruginosa_UFPEDA261", "target": "Pseudomonas aeruginosa UFPEDA261", "target_type": "Gram-", "surface_charge": -0.90, "anionic_fraction": 0.65, "lps": 1, "peptidoglycan": 1, "teichoic_acid": 0, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
    {"target_id": "P_aeruginosa_UFPEDA262", "target": "Pseudomonas aeruginosa UFPEDA262", "target_type": "Gram-", "surface_charge": -0.90, "anionic_fraction": 0.65, "lps": 1, "peptidoglycan": 1, "teichoic_acid": 0, "cholesterol": 0, "ergosterol": 0, "viral_envelope": 0},
]


def load_project_targets() -> pd.DataFrame:
    project = pd.read_parquet(ROOT / "data" / "processed" / "project_membrane_targets.parquet")
    literature = pd.DataFrame(LITERATURE_TARGETS)
    frames = [project, literature]
    bench = load_bench_targets()
    if not bench.empty:
        frames.append(bench)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["target_id"])


def build_pairs() -> pd.DataFrame:
    peptides = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base_project.parquet")
    targets = load_project_targets()

    rows = []
    for _, pep in peptides.iterrows():
        if not pep.get("sequence"):
            continue
        for _, mem in targets.iterrows():
            q_p = peptide_q(pep)
            h_p = peptide_h(pep)
            mu_p = peptide_mu_h(pep)
            q_m = float(mem["surface_charge"])
            h_m = float(mem.get("hydrophobicity", mem.get("hydrophobicity_computed", 0.5)) or 0.5)
            chol = float(mem.get("cholesterol", 0) or 0)

            pmi = compute_pmi(q_p, q_m, h_p, h_m, mu_p, chol)

            rows.append(
                {
                    "peptide_id": pep["peptide_id"],
                    "sequence": pep["sequence"],
                    "target_id": mem["target_id"],
                    "target": mem["target"],
                    "target_type": mem["target_type"],
                    "q_peptide": q_p,
                    "h_peptide": h_p,
                    "mu_h_peptide": mu_p,
                    "surface_charge": q_m,
                    "anionic_fraction": mem.get("anionic_fraction"),
                    "cholesterol": chol,
                    "lps": mem.get("lps"),
                    "peptidoglycan": mem.get("peptidoglycan"),
                    "ergosterol": mem.get("ergosterol"),
                    "viral_envelope": mem.get("viral_envelope"),
                    "pmi": pmi,
                }
            )

    pairs = pd.DataFrame(rows)

    # Seletividade vs célula normal
    normal = pairs[pairs["target_id"] == "cell_normal"].set_index("peptide_id")["pmi"]
    pairs["pmi_normal"] = pairs["peptide_id"].map(normal)
    pairs["pmi_sel"] = pairs.apply(
        lambda r: compute_pmi_sel(r["pmi"], r["pmi_normal"]) if pd.notna(r["pmi_normal"]) else np.nan,
        axis=1,
    )

    # Anexar endpoints experimentais (MIC/MBC/CC50/hemólise/…)
    ep = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_endpoints.parquet")
    ep = ep.dropna(subset=["endpoint", "value"])
    for endpoint_name in ("MIC", "MBC", "CC50", "IC50", "HEMOLYSIS", "BIOFILM_INHIB"):
        sub = ep[ep["endpoint"] == endpoint_name][
            ["peptide_id", "target_id", "value", "unit", "source", "confidence"]
        ]
        if sub.empty:
            continue
        prefix = endpoint_name.lower()
        sub = sub.rename(
            columns={
                "value": f"{prefix}_value",
                "unit": f"{prefix}_unit",
                "source": f"{prefix}_source",
                "confidence": f"{prefix}_confidence",
            }
        )
        pairs = pairs.merge(sub, on=["peptide_id", "target_id"], how="left")

    out = ROOT / "data" / "processed"
    pairs.to_parquet(out / "pepmem_pairs.parquet", index=False)
    pairs.to_csv(out / "pepmem_pairs.csv", index=False)
    return pairs


def main() -> None:
    pairs = build_pairs()
    summary = {
        "pairs_rows": len(pairs),
        "with_mic": int(pairs["mic_value"].notna().sum()) if "mic_value" in pairs.columns else 0,
        "with_hemolysis": int(pairs["hemolysis_value"].notna().sum())
        if "hemolysis_value" in pairs.columns
        else 0,
    }
    out = ROOT / "data" / "processed" / "pairs_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Pares peptídeo-membrana:", summary)


if __name__ == "__main__":
    main()
