#!/usr/bin/env python3
"""Import lab MIC/MBC data from data/bench/ and retrain models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_mic import BENCH_DIR, load_bench_mic


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def retrain_pipeline(new_peptides: bool) -> None:
    run([sys.executable, str(ROOT / "scripts" / "build_pairs.py")])
    if new_peptides:
        print("Novos peptídeos detectados — gerando embeddings faltantes (ESM-2)...")
        run([
            sys.executable,
            str(ROOT / "scripts" / "generate_embeddings.py"),
            "--scope",
            "project",
            "--missing-only",
        ])
    run([sys.executable, str(ROOT / "scripts" / "train_baseline.py")])
    run([sys.executable, str(ROOT / "scripts" / "train_multimodal.py")])
    run([sys.executable, str(ROOT / "scripts" / "compute_shap.py")])


def check_only() -> None:
    mic_df = load_bench_mic()
    if mic_df.empty:
        print("OK: mic_bench.csv sem linhas de dados (só cabeçalho).")
        return
    print(f"OK: {len(mic_df)} linhas válidas")
    print(mic_df[["peptide_id", "sequence", "target_id", "endpoint", "value"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Importar MICs da bancada (data/bench/)")
    parser.add_argument("--check", action="store_true", help="Só valida mic_bench.csv")
    parser.add_argument("--retrain", action="store_true", help="Retreina modelos e SHAP")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_only()
        return

    mic_df = load_bench_mic()
    if mic_df.empty:
        print("Preencha data/bench/mic_bench.csv com seus MICs e rode novamente.")
        print("Documentação: data/bench/README.md")
        return

    before_ids = set()
    proj_path = ROOT / "data" / "processed" / "pepmem_base_project.parquet"
    if proj_path.exists():
        import pandas as pd

        before_ids = set(pd.read_parquet(proj_path)["peptide_id"].astype(str))

    run([sys.executable, str(ROOT / "scripts" / "build_datasets.py")])
    run([sys.executable, str(ROOT / "scripts" / "build_pairs.py")])

    after_ids = set()
    if proj_path.exists():
        import pandas as pd

        after_ids = set(pd.read_parquet(proj_path)["peptide_id"].astype(str))
    new_peptides = bool(after_ids - before_ids)

    pairs = json.loads((ROOT / "data" / "processed" / "pairs_summary.json").read_text(encoding="utf-8"))
    summary_path = ROOT / "data" / "processed" / "build_summary.json"
    build_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report = {
        "bench_mic_rows": build_summary.get("pepmem_endpoints_bench_rows", len(mic_df)),
        "training_mic_pairs": pairs.get("with_mic"),
        "new_peptide_ids": sorted(after_ids - before_ids),
    }
    (BENCH_DIR / "import_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMICs no treino: {pairs.get('with_mic', '?')}")

    if args.retrain:
        retrain_pipeline(new_peptides)
    else:
        print("\nModelos ainda não retreinados. Rode:")
        print("  python scripts/import_bench_mic.py --retrain")


if __name__ == "__main__":
    main()
