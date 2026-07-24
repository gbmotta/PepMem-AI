#!/usr/bin/env python3
"""Baixa listas FASTA do APD6 (Antimicrobial Peptide Database).

Entrada: API pública em aps.unmc.edu. Saída: FASTAs e ``manifest.json`` em
``data/raw/apd/``.

Papel no pipeline: ingestão de AMPs de referência — precede ``build_datasets.py``.

Execução:
    python scripts/download_apd.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://aps.unmc.edu/assets/sequences"
FILES = [
    "naturalAMPs_APD2024a.fasta",
    "animalAMPs_APD2024a.fasta",
    "plantAMPs_APD2024.fasta",
    "bacteriaAMPs_APD2024.fasta",
    "humanAMPs_APD2024.fasta",
    "amphibianAMPs_APD2024.fasta",
    "insectAMPs_APD2024.fasta",
]


def main() -> None:
    """Baixa todos os FASTAs configurados e grava manifesto com contagens."""
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw" / "apd"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "PepMem-AI/1.0"

    manifest: dict[str, int | str] = {
        "source": BASE,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for name in FILES:
        url = f"{BASE}/{name}"
        dest = out_dir / name
        print(f"Baixando {name}...")
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        count = resp.text.count("\n>")
        manifest[name] = count

    save = out_dir / "manifest.json"
    save.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"APD salvo em {out_dir}")


if __name__ == "__main__":
    main()
