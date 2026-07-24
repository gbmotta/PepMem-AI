"""Caminhos do projeto PepMem-AI (local e Spaces).

Centraliza a resolução da raiz do repositório para que scripts, dashboard e
artefatos em ``data/processed/`` usem o mesmo ponto de partida.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Retorna a raiz do repositório (pasta que contém ``pepmem/`` e ``data/``)."""
    return Path(__file__).resolve().parent.parent
