#!/usr/bin/env python3
"""Executa o pipeline PepMem-AI de ponta a ponta.

Etapas: build_datasets → build_pairs → embeddings → treino baseline/multimodal → SHAP.

Papel no pipeline: orquestrador principal após download de OPM/APD.

Execução:
    python scripts/run_pipeline.py

Pré-requisitos: ``data/raw/opm/`` (``download_opm.py``); APD opcional.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> None:
    """Executa um passo do pipeline no diretório raiz do projeto."""
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    """Roda sequencialmente datasets, pares, embeddings, treino e SHAP."""
    steps = [
        [sys.executable, str(SCRIPTS / "build_datasets.py")],
        [sys.executable, str(SCRIPTS / "build_pairs.py")],
        [sys.executable, str(SCRIPTS / "generate_embeddings.py"), "--scope", "all"],
        [sys.executable, str(SCRIPTS / "train_baseline.py")],
        [sys.executable, str(SCRIPTS / "train_multimodal.py")],
        [sys.executable, str(SCRIPTS / "compute_shap.py")],
    ]
    for step in steps:
        run(step)
    print("\nPipeline concluído.")


if __name__ == "__main__":
    main()
