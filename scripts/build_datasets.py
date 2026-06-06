#!/usr/bin/env python3
"""Build PepMem datasets from raw OPM data and project seed records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peptide_utils import add_descriptor_columns, parse_fasta

# PepMem target types used in experimental validation (from project docs)
PROJECT_TARGETS = [
    {
        "target_id": "S_aureus_ATCC29213",
        "target": "Staphylococcus aureus ATCC 29213",
        "target_type": "Gram+",
        "surface_charge": -0.80,
        "anionic_fraction": 0.60,
        "lps": 0,
        "peptidoglycan": 1,
        "teichoic_acid": 1,
        "cholesterol": 0,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 8,
        "opm_membrane_name": "Bacterial Gram-positive plasma membrane",
    },
    {
        "target_id": "S_epidermidis_ATCC12228",
        "target": "Staphylococcus epidermidis ATCC 12228",
        "target_type": "Gram+",
        "surface_charge": -0.80,
        "anionic_fraction": 0.60,
        "lps": 0,
        "peptidoglycan": 1,
        "teichoic_acid": 1,
        "cholesterol": 0,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 8,
        "opm_membrane_name": "Bacterial Gram-positive plasma membrane",
    },
    {
        "target_id": "E_coli_ATCC25922",
        "target": "Escherichia coli ATCC 25922",
        "target_type": "Gram-",
        "surface_charge": -0.90,
        "anionic_fraction": 0.65,
        "lps": 1,
        "peptidoglycan": 1,
        "teichoic_acid": 0,
        "cholesterol": 0,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 2,
        "opm_membrane_name": "Bacterial Gram-negative inner membrane",
    },
    {
        "target_id": "P_aeruginosa_ATCC27853",
        "target": "Pseudomonas aeruginosa ATCC 27853",
        "target_type": "Gram-",
        "surface_charge": -0.90,
        "anionic_fraction": 0.65,
        "lps": 1,
        "peptidoglycan": 1,
        "teichoic_acid": 0,
        "cholesterol": 0,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 2,
        "opm_membrane_name": "Bacterial Gram-negative inner membrane",
    },
    {
        "target_id": "Candida_spp",
        "target": "Candida spp.",
        "target_type": "fungo",
        "surface_charge": -0.40,
        "anionic_fraction": 0.25,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 0,
        "ergosterol": 1,
        "viral_envelope": 0,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
    {
        "target_id": "T_cruzi",
        "target": "Trypanosoma cruzi",
        "target_type": "parasita",
        "surface_charge": -0.50,
        "anionic_fraction": 0.30,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 0,
        "ergosterol": 1,
        "viral_envelope": 0,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
    {
        "target_id": "Zika_PE243",
        "target": "Zika virus PE243",
        "target_type": "vírus envelopado",
        "surface_charge": -0.30,
        "anionic_fraction": 0.20,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 1,
        "ergosterol": 0,
        "viral_envelope": 1,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
    {
        "target_id": "HSV1",
        "target": "Herpes simplex virus type 1",
        "target_type": "vírus envelopado",
        "surface_charge": -0.30,
        "anionic_fraction": 0.20,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 1,
        "ergosterol": 0,
        "viral_envelope": 1,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
    {
        "target_id": "cell_normal",
        "target": "célula mamífera normal",
        "target_type": "mamífero normal",
        "surface_charge": -0.10,
        "anionic_fraction": 0.10,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 1,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
    {
        "target_id": "cell_tumor",
        "target": "célula tumoral",
        "target_type": "célula tumoral",
        "surface_charge": -0.40,
        "anionic_fraction": 0.25,
        "lps": 0,
        "peptidoglycan": 0,
        "teichoic_acid": 0,
        "cholesterol": 1,
        "ergosterol": 0,
        "viral_envelope": 0,
        "opm_membrane_id": 4,
        "opm_membrane_name": "Eukaryotic plasma membrane",
    },
]

# Project peptides from CNPq proposal (Quadro 01); * = patented substitutions (sequence alias kept)
PROJECT_PEPTIDES = [
    {"peptide_id": "P01", "name": "Stigmurin_analog_1", "parent": "Stigmurin", "sequence_raw": "FFSLIP*LV*GLISAFK", "sequence": "FFSLIPSLVGGLISAFK", "net_charge": 3, "molecular_mass_kda": 1.90, "hydrophobicity": 0.78, "hydrophobic_moment": 0.67, "isoelectric_point": 9.54},
    {"peptide_id": "P02", "name": "Stigmurin_analog_2", "parent": "Stigmurin", "sequence_raw": "FFSLIP*LVG*LISAFK", "sequence": "FFSLIPSLVGLISAFK", "net_charge": 4, "molecular_mass_kda": 1.90, "hydrophobicity": 0.78, "hydrophobic_moment": 0.65, "isoelectric_point": 9.54},
    {"peptide_id": "P03", "name": "Stigmurin_analog_3", "parent": "Stigmurin", "sequence_raw": "FF*LIP*LV*GLISAFK", "sequence": "FFSLIPSLVGGLISAFK", "net_charge": 4, "molecular_mass_kda": 1.95, "hydrophobicity": 0.72, "hydrophobic_moment": 0.72, "isoelectric_point": 9.66},
    {"peptide_id": "P04", "name": "Stigmurin_analog_4", "parent": "Stigmurin", "sequence_raw": "FFSLIP*LVG*LI*AFK", "sequence": "FFSLIPSLVGLISAFK", "net_charge": 5, "molecular_mass_kda": 1.95, "hydrophobicity": 0.72, "hydrophobic_moment": 0.70, "isoelectric_point": 9.66},
    {"peptide_id": "P05", "name": "TsAP2_native", "parent": "TsAP-2", "sequence_raw": "FLGMIPGLIGGLISAFK", "sequence": "FLGMIPGLIGGLISAFK", "net_charge": 2, "molecular_mass_kda": 1.73, "hydrophobicity": 0.90, "hydrophobic_moment": 0.59, "isoelectric_point": 9.07},
    {"peptide_id": "P06", "name": "TsAP2_analog_1", "parent": "TsAP-2", "sequence_raw": "FL*MIPGLI*GLI*AF*", "sequence": "FLGMIPGLIGGLISAFK", "net_charge": 5, "molecular_mass_kda": 2.03, "hydrophobicity": 0.72, "hydrophobic_moment": 0.75, "isoelectric_point": 12.32},
    {"peptide_id": "P07", "name": "TsAP2_analog_2", "parent": "TsAP-2", "sequence_raw": "FL*MIP*LIGGLI*AF*", "sequence": "FLGMIPGLIGGLISAFK", "net_charge": 3, "molecular_mass_kda": 1.90, "hydrophobicity": 0.78, "hydrophobic_moment": 0.70, "isoelectric_point": 11.84},
    {"peptide_id": "P08", "name": "TsAP2_analog_3", "parent": "TsAP-2", "sequence_raw": "FLGMIP*LI*GLI*AF*", "sequence": "FLGMIPGLIGGLISAFK", "net_charge": 5, "molecular_mass_kda": 2.00, "hydrophobicity": 0.78, "hydrophobic_moment": 0.70, "isoelectric_point": 12.14},
    {"peptide_id": "P09", "name": "TsAP2_analog_4", "parent": "TsAP-2", "sequence_raw": "FL*MIP*LI**LISAF*", "sequence": "FLGMIPGLIGGLISAFK", "net_charge": 6, "molecular_mass_kda": 2.07, "hydrophobicity": 0.66, "hydrophobic_moment": 0.77, "isoelectric_point": 11.85},
    {"peptide_id": "P10", "name": "Stigmurin_native", "parent": "Stigmurin", "apd_id": "AP02531", "sequence_raw": "FFSLIPSLVGGLISAFK", "sequence": "FFSLIPSLVGGLISAFK", "net_charge": 3, "molecular_mass_kda": 1.90, "hydrophobicity": 0.78, "hydrophobic_moment": 0.67, "isoelectric_point": 9.54},
    {"peptide_id": "P11", "name": "StigA6", "parent": "Stigmurin", "sequence_raw": "FFSLIPKLVKGLISAFK", "sequence": "FFSLIPKLVKGLISAFK", "net_charge": 3, "source": "Parente_2022_thesis"},
    {"peptide_id": "P12", "name": "StigA16", "parent": "Stigmurin", "sequence_raw": "FFKLIPKLVKGLISAFK", "sequence": "FFKLIPKLVKGLISAFK", "net_charge": 4, "source": "Parente_2022_thesis"},
]

# MIC/MBC from Parente 2022, Table 3 (multidrug-resistant clinical strains)
LITERATURE_ENDPOINTS = [
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1040", "target": "Staphylococcus aureus UFPEDA1040", "target_type": "Gram+", "endpoint": "MIC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1040", "target": "Staphylococcus aureus UFPEDA1040", "target_type": "Gram+", "endpoint": "MBC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1040", "target": "Staphylococcus aureus UFPEDA1040", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1040", "target": "Staphylococcus aureus UFPEDA1040", "target_type": "Gram+", "endpoint": "MBC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1051", "target": "Staphylococcus aureus UFPEDA1051", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1051", "target": "Staphylococcus aureus UFPEDA1051", "target_type": "Gram+", "endpoint": "MBC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1051", "target": "Staphylococcus aureus UFPEDA1051", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1051", "target": "Staphylococcus aureus UFPEDA1051", "target_type": "Gram+", "endpoint": "MBC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1058", "target": "Staphylococcus aureus UFPEDA1058", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1058", "target": "Staphylococcus aureus UFPEDA1058", "target_type": "Gram+", "endpoint": "MBC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1058", "target": "Staphylococcus aureus UFPEDA1058", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1058", "target": "Staphylococcus aureus UFPEDA1058", "target_type": "Gram+", "endpoint": "MBC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1059", "target": "Staphylococcus aureus UFPEDA1059", "target_type": "Gram+", "endpoint": "MIC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "S_aureus_UFPEDA1059", "target": "Staphylococcus aureus UFPEDA1059", "target_type": "Gram+", "endpoint": "MBC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1059", "target": "Staphylococcus aureus UFPEDA1059", "target_type": "Gram+", "endpoint": "MIC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "S_aureus_UFPEDA1059", "target": "Staphylococcus aureus UFPEDA1059", "target_type": "Gram+", "endpoint": "MBC", "value": 2.3, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "P_aeruginosa_UFPEDA261", "target": "Pseudomonas aeruginosa UFPEDA261", "target_type": "Gram-", "endpoint": "MIC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "P_aeruginosa_UFPEDA261", "target": "Pseudomonas aeruginosa UFPEDA261", "target_type": "Gram-", "endpoint": "MBC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "P_aeruginosa_UFPEDA261", "target": "Pseudomonas aeruginosa UFPEDA261", "target_type": "Gram-", "endpoint": "MIC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "P_aeruginosa_UFPEDA261", "target": "Pseudomonas aeruginosa UFPEDA261", "target_type": "Gram-", "endpoint": "MBC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "P_aeruginosa_UFPEDA262", "target": "Pseudomonas aeruginosa UFPEDA262", "target_type": "Gram-", "endpoint": "MIC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P11", "target_id": "P_aeruginosa_UFPEDA262", "target": "Pseudomonas aeruginosa UFPEDA262", "target_type": "Gram-", "endpoint": "MBC", "value": 4.7, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "P_aeruginosa_UFPEDA262", "target": "Pseudomonas aeruginosa UFPEDA262", "target_type": "Gram-", "endpoint": "MIC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
    {"peptide_id": "P12", "target_id": "P_aeruginosa_UFPEDA262", "target": "Pseudomonas aeruginosa UFPEDA262", "target_type": "Gram-", "endpoint": "MBC", "value": 3.4, "unit": "uM", "reference": "Parente_2022_Table3"},
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_opm_membranes_df(opm_dir: Path) -> pd.DataFrame:
    membranes = load_json(opm_dir / "membranes.json")
    df = pd.DataFrame(membranes)
    df["source"] = "OPM"
    df["dataset"] = "Membrane-Targets-OPM"
    return df


def build_opm_proteins_df(opm_dir: Path) -> pd.DataFrame:
    proteins = load_json(opm_dir / "primary_structures.json")
    df = pd.DataFrame(proteins)
    if "uniprotcodes" in df.columns:
        df["uniprotcodes"] = df["uniprotcodes"].apply(lambda x: ";".join(x) if isinstance(x, list) else x)
    df["source"] = "OPM"
    return df


def map_opm_to_target_type(name: str) -> str:
    name_l = name.lower()
    if "gram-negative" in name_l:
        return "Gram-"
    if "gram-positive" in name_l:
        return "Gram+"
    if "mitochond" in name_l:
        return "organelle"
    if "archae" in name_l:
        return "arquea"
    if "viral" in name_l or "envelope" in name_l:
        return "vírus/membrana"
    if "eukaryotic" in name_l or "endoplasm" in name_l or "golgi" in name_l:
        return "eucariota"
    return "outro"


def build_membrane_targets(opm_dir: Path, out_dir: Path) -> pd.DataFrame:
    opm_mem = build_opm_membranes_df(opm_dir)
    opm_mem["target_type_mapped"] = opm_mem["name"].map(map_opm_to_target_type)

    project_df = pd.DataFrame(PROJECT_TARGETS)
    project_df["source"] = "PepMem_project"
    project_df["dataset"] = "Membrane-Targets-Project"

    combined = pd.concat(
        [
            opm_mem.rename(columns={"id": "opm_id", "name": "membrane_name"}),
            project_df,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_parquet(out_dir / "membrane_targets.parquet", index=False)
    combined.to_csv(out_dir / "membrane_targets.csv", index=False)
    opm_mem.to_parquet(out_dir / "opm_membranes.parquet", index=False)
    project_df.to_parquet(out_dir / "project_membrane_targets.parquet", index=False)
    return combined


def build_apd_base(apd_dir: Path) -> pd.DataFrame:
    fasta_path = apd_dir / "naturalAMPs_APD2024a.fasta"
    if not fasta_path.exists():
        return pd.DataFrame()

    records = parse_fasta(fasta_path)
    rows = []
    for rec in records:
        if not rec.get("apd_id") or not rec.get("sequence"):
            continue
        rows.append(
            {
                "peptide_id": f"APD:{rec['apd_id']}",
                "apd_id": rec["apd_id"],
                "name": rec["apd_id"],
                "parent": None,
                "sequence_raw": None,
                "sequence": rec["sequence"],
                "c_terminal": None,
                "source": "APD2024_natural",
                "dataset": "PepMem-Base-APD",
                "embedding_esm2": None,
                "embedding_protbert": None,
            }
        )

    df = pd.DataFrame(rows)
    return add_descriptor_columns(df)


def build_pepmem_base(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_rows = []
    for pep in PROJECT_PEPTIDES:
        project_rows.append(
            {
                **pep,
                "c_terminal": "NH2",
                "source": pep.get("source", "CNPq_proposal_2025"),
                "dataset": "PepMem-Base-Project",
                "embedding_esm2": None,
                "embedding_protbert": None,
            }
        )

    project_df = add_descriptor_columns(pd.DataFrame(project_rows))
    project_df.to_parquet(out_dir / "pepmem_base_project.parquet", index=False)
    project_df.to_csv(out_dir / "pepmem_base_project.csv", index=False)

    apd_df = build_apd_base(root / "data" / "raw" / "apd")
    if not apd_df.empty:
        project_seqs = set(project_df["sequence"].dropna())
        apd_df = apd_df[~apd_df["sequence"].isin(project_seqs)].copy()
        apd_df.to_parquet(out_dir / "pepmem_base_apd.parquet", index=False)
        apd_df.to_csv(out_dir / "pepmem_base_apd.csv", index=False)

    full_df = pd.concat([project_df, apd_df], ignore_index=True, sort=False)
    full_df.to_parquet(out_dir / "pepmem_base.parquet", index=False)
    full_df.to_csv(out_dir / "pepmem_base.csv", index=False)
    return project_df, apd_df, full_df


def build_pepmem_endpoints(project_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    project_peptides = project_df[project_df["peptide_id"].str.match(r"^P\d+$") & project_df["sequence"].notna()]
    project_targets = pd.DataFrame(PROJECT_TARGETS)

    rows = []
    for _, pep in project_peptides.iterrows():
        for _, tgt in project_targets.iterrows():
            rows.append(
                {
                    "peptide_id": pep["peptide_id"],
                    "sequence": pep["sequence"],
                    "target_id": tgt["target_id"],
                    "target": tgt["target"],
                    "target_type": tgt["target_type"],
                    "endpoint": None,
                    "value": None,
                    "unit": None,
                    "assay": "microdilution",
                    "reference": None,
                    "source": "scaffold",
                    "confidence": "pending",
                    "split": None,
                }
            )

    scaffold_df = pd.DataFrame(rows)

    lit_rows = []
    seq_map = project_df.set_index("peptide_id")["sequence"].to_dict()
    for item in LITERATURE_ENDPOINTS:
        lit_rows.append(
            {
                **item,
                "sequence": seq_map.get(item["peptide_id"]),
                "assay": "microdilution",
                "source": "literature",
                "confidence": "experimental",
                "split": "literature",
            }
        )
    literature_df = pd.DataFrame(lit_rows)

    endpoints_df = pd.concat([scaffold_df, literature_df], ignore_index=True, sort=False)
    endpoints_df.to_parquet(out_dir / "pepmem_endpoints.parquet", index=False)
    endpoints_df.to_csv(out_dir / "pepmem_endpoints.csv", index=False)
    literature_df.to_parquet(out_dir / "pepmem_endpoints_literature.parquet", index=False)
    return endpoints_df


def build_opm_reference_tables(opm_dir: Path, out_dir: Path) -> None:
    """Flatten key OPM tables for downstream feature engineering."""
    proteins = build_opm_proteins_df(opm_dir)
    proteins.to_parquet(out_dir / "opm_primary_structures.parquet", index=False)
    proteins.to_csv(out_dir / "opm_primary_structures.csv", index=False)

    for name in ("species", "families", "superfamilies", "assemblies", "membranes"):
        path = opm_dir / f"{name}.json"
        if path.exists():
            pd.DataFrame(load_json(path)).to_parquet(out_dir / f"opm_{name}.parquet", index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    opm_dir = root / "data" / "raw" / "opm"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = opm_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Execute primeiro: python scripts/download_opm.py\nOPM não encontrado em {opm_dir}")

    apd_dir = root / "data" / "raw" / "apd"
    if not (apd_dir / "naturalAMPs_APD2024a.fasta").exists():
        print("AVISO: APD não encontrado. Execute: python scripts/download_apd.py")

    print("Construindo datasets...")
    membrane_df = build_membrane_targets(opm_dir, out_dir)
    project_df, apd_df, peptides_df = build_pepmem_base(root, out_dir)
    endpoints_df = build_pepmem_endpoints(project_df, out_dir)
    build_opm_reference_tables(opm_dir, out_dir)

    summary = {
        "membrane_targets_rows": len(membrane_df),
        "pepmem_base_project_rows": len(project_df),
        "pepmem_base_apd_rows": len(apd_df),
        "pepmem_base_total_rows": len(peptides_df),
        "pepmem_endpoints_rows": len(endpoints_df),
        "pepmem_endpoints_literature_rows": len(LITERATURE_ENDPOINTS),
        "opm_primary_structures": len(load_json(opm_dir / "primary_structures.json")),
    }
    with (out_dir / "build_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Datasets gerados em", out_dir)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
