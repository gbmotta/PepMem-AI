#!/usr/bin/env python3
"""Converte docs/*.md (TREINO, INTERPRETACAO_RESULTADOS, DEPLOY) em DOCX e PDF.

Requer: pypandoc-binary (ou pandoc no PATH) e LibreOffice (soffice).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "exports"
DOCS_NAMES = ("TREINO", "INTERPRETACAO_RESULTADOS", "DEPLOY")


def _pandoc() -> str:
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except Exception:
        return "pandoc"


def convert_one(name: str, pandoc: str) -> None:
    md = DOCS / f"{name}.md"
    if not md.exists():
        raise FileNotFoundError(md)
    docx = OUT / f"{name}.docx"
    pdf = OUT / f"{name}.pdf"

    print(f"→ {name}.docx")
    subprocess.run(
        [pandoc, str(md), "-o", str(docx), "--from", "markdown", "--to", "docx", f"--resource-path={DOCS}"],
        check=True,
    )

    print(f"→ {name}.pdf (via LibreOffice)")
    profile = Path("/tmp/lo_profile_pepmem")
    profile.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HOME", str(Path.home()))
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--norestore",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(OUT),
            str(docx),
        ],
        check=True,
        env=env,
    )
    if not pdf.exists():
        raise RuntimeError(f"PDF não gerado: {pdf}")
    print(f"  OK  {docx.stat().st_size // 1024} KB docx · {pdf.stat().st_size // 1024} KB pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pandoc = _pandoc()
    print(f"pandoc: {pandoc}")
    for name in DOCS_NAMES:
        convert_one(name, pandoc)
    print(f"\nArquivos em {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
