#!/usr/bin/env python3
"""Baixa o banco OPM completo via API REST pública.

Entrada: ``https://opm-back.cc.lehigh.edu/opm-backend``. Saída: JSONs paginados
e ``manifest.json`` em ``data/raw/opm/``.

Papel no pipeline: ingestão de alvos de membrana — precede ``build_datasets.py``.

Execução:
    python scripts/download_opm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

BASE_URL = "https://opm-back.cc.lehigh.edu/opm-backend"
PAGE_SIZE = 50
REQUEST_DELAY = 0.05

# --- endpoints paginados (objects + total_objects) ---
PAGINATED = [
    "primary_structures",
    "structure_subunits",
    "species",
    "families",
    "superfamilies",
    "classtypes",
    "types",
    "assemblies",
    "assembly_families",
    "assembly_superfamilies",
    "assembly_membranes",
    "citations",
    "membranes",
]

# --- endpoints estáticos (resposta única) ---
STATIC = ["stats", "pdbids"]


def fetch_page(session: requests.Session, endpoint: str, page_num: int) -> dict[str, Any]:
    """Busca uma página da API OPM com retentativas exponenciais."""
    url = f"{BASE_URL}/{endpoint}"
    params = {"page_num": page_num, "page_size": PAGE_SIZE}
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
            last_exc = exc
    raise last_exc  # type: ignore[possibly-undefined]


def fetch_all_paginated(session: requests.Session, endpoint: str) -> list[dict[str, Any]]:
    """Percorre todas as páginas de um endpoint e retorna a lista completa."""
    first = fetch_page(session, endpoint, 1)
    total = first["total_objects"]
    objects = list(first["objects"])
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    if pages <= 1:
        return objects

    for page in tqdm(range(2, pages + 1), desc=endpoint, unit="page"):
        data = fetch_page(session, endpoint, page)
        objects.extend(data["objects"])
        time.sleep(REQUEST_DELAY)

    if len(objects) != total:
        print(f"  aviso: {endpoint} esperado {total}, obtido {len(objects)}")
    return objects


def fetch_static(session: requests.Session, endpoint: str) -> Any:
    """Busca endpoint não paginado (stats, pdbids)."""
    url = f"{BASE_URL}/{endpoint}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def save_json(path: Path, data: Any) -> None:
    """Grava dados em JSON com indentação e UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    """Baixa endpoints estáticos e paginados; grava manifesto com contagens."""
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw" / "opm"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "PepMem-AI/1.0"})

    print(f"Salvando dados OPM em {out_dir}")

    for endpoint in STATIC:
        print(f"Baixando {endpoint}...")
        data = fetch_static(session, endpoint)
        save_json(out_dir / f"{endpoint}.json", data)

    manifest: dict[str, int | str] = {"source": BASE_URL, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for endpoint in PAGINATED:
        print(f"Baixando {endpoint}...")
        records = fetch_all_paginated(session, endpoint)
        save_json(out_dir / f"{endpoint}.json", records)
        manifest[endpoint] = len(records)

    save_json(out_dir / "manifest.json", manifest)
    print("\nDownload OPM concluído:")
    for key, val in manifest.items():
        if key not in ("source", "downloaded_at"):
            print(f"  {key}: {val} registros")


if __name__ == "__main__":
    main()
