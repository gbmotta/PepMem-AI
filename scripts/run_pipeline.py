#!/usr/bin/env python3
"""Run the PepMem-AI data pipeline end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
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
