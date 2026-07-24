"""Reexporta ``pepmem.narrative`` carregando pelo caminho do arquivo.

Fica em ``dashboard/`` para o Streamlit Cloud sempre resolver o módulo ao lado
de ``app.py``, sem duplicar templates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "pepmem" / "narrative.py"
_SPEC = importlib.util.spec_from_file_location("pepmem_narrative_dash", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Não foi possível carregar narrativa em {_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

llm_status = _MOD.llm_status
narrate_batch = _MOD.narrate_batch
narrate_ranking = _MOD.narrate_ranking
narrate_shap_overview = _MOD.narrate_shap_overview
narrate_single = _MOD.narrate_single

__all__ = [
    "llm_status",
    "narrate_batch",
    "narrate_ranking",
    "narrate_shap_overview",
    "narrate_single",
]
