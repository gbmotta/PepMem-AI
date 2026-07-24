"""Narrativa em português dos resultados PepMem-AI (sem alterar predições).

Template sempre disponível. Qwen GGUF opcional via llama-cpp-python se o
arquivo existir (ou download opcional com PEPMEM_LLM_AUTO_DOWNLOAD=1).

Papel: só explicar KPIs/PMI/vizinhos/SHAP já calculados pelo RF.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from pepmem.paths import project_root

ROOT = project_root()
MODELS_DIR = ROOT / "models" / "llm"

# Qwen 0.5B Instruct GGUF (pequeno o bastante para demo em CPU)
DEFAULT_GGUF_REPO = os.environ.get(
    "PEPMEM_GGUF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
)
DEFAULT_GGUF_FILE = os.environ.get(
    "PEPMEM_GGUF_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

_llm_lock = threading.Lock()
_llm = None
_llm_path: Path | None = None
_llm_error: str | None = None

SYSTEM_PROMPT = (
    "Você é um assistente do PepMem-AI (InovAI Lab/UFRN). "
    "Explique resultados de predição peptídeo–membrana em português claro, "
    "para priorização in vitro. "
    "Use APENAS os números e fatos do contexto. "
    "NÃO invente MIC, probabilidade ou PMI. "
    "NÃO altere nem recalcule resultados. "
    "Termine lembrando que é priorização, não substitui ensaio de bancada. "
    "Responda em 1 curto parágrafo (4–7 frases)."
)


def llm_status() -> dict[str, Any]:
    """Estado do motor narrativo (template vs GGUF)."""
    path = resolve_gguf_path(download=False)
    has_llama = False
    try:
        import llama_cpp  # noqa: F401

        has_llama = True
    except Exception:
        has_llama = False
    return {
        "template_always": True,
        "llama_cpp_installed": has_llama,
        "gguf_path": str(path) if path else None,
        "gguf_ready": bool(path and path.is_file()),
        "auto_download": os.environ.get("PEPMEM_LLM_AUTO_DOWNLOAD", "").strip() in {"1", "true", "yes"},
        "last_error": _llm_error,
    }


def resolve_gguf_path(*, download: bool = False) -> Path | None:
    """Localiza GGUF: PEPMEM_GGUF_PATH → models/llm → download opcional."""
    env = os.environ.get("PEPMEM_GGUF_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local = MODELS_DIR / DEFAULT_GGUF_FILE
    if local.is_file():
        return local
    # qualquer .gguf na pasta
    found = sorted(MODELS_DIR.glob("*.gguf"))
    if found:
        return found[0]

    if not download:
        return None
    if os.environ.get("PEPMEM_LLM_AUTO_DOWNLOAD", "").strip() not in {"1", "true", "yes"}:
        return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DEFAULT_GGUF_REPO,
            filename=DEFAULT_GGUF_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:
        global _llm_error
        _llm_error = f"Download GGUF falhou: {e}"
        return None


def _get_llm():
    """Carrega llama-cpp uma vez (lazy). Retorna (model|None, erro|None)."""
    global _llm, _llm_path, _llm_error
    with _llm_lock:
        if _llm is not None:
            return _llm, None
        try:
            from llama_cpp import Llama
        except Exception as e:
            _llm_error = f"llama-cpp-python ausente: {e}"
            return None, _llm_error

        path = resolve_gguf_path(download=True)
        if path is None or not path.is_file():
            _llm_error = (
                "GGUF não encontrado. Coloque em models/llm/ ou defina "
                "PEPMEM_GGUF_PATH (opcional: PEPMEM_LLM_AUTO_DOWNLOAD=1)."
            )
            return None, _llm_error

        try:
            _llm = Llama(
                model_path=str(path),
                n_ctx=2048,
                n_threads=max(1, (os.cpu_count() or 2) // 2),
                verbose=False,
            )
            _llm_path = path
            _llm_error = None
            return _llm, None
        except Exception as e:
            _llm_error = f"Falha ao carregar GGUF: {e}"
            return None, _llm_error


def _band_pt(prob: float) -> str:
    if prob >= 0.70:
        return "alta confiança de atividade (priorizar ensaio)"
    if prob >= 0.40:
        return "candidato intermediário"
    return "baixa probabilidade de atividade"


def template_explain_single(
    *,
    sequence: str,
    target_label: str,
    prob: float,
    pmi: float | None = None,
    interval: str | None = None,
    q_peptide: float | None = None,
    neighbors: list[dict[str, Any]] | None = None,
    shap_top: list[dict[str, Any]] | None = None,
    in_project: bool = False,
) -> str:
    """Narrativa determinística a partir dos números já calculados."""
    parts: list[str] = []
    seq_short = sequence if len(sequence) <= 24 else f"{sequence[:21]}…"
    parts.append(
        f"Para o peptídeo {seq_short} contra {target_label}, o modelo calibrou "
        f"{prob:.0%} de chance de alta atividade (MIC ≤ 3,4 µM) — faixa: {_band_pt(prob)}."
    )
    if pmi is not None:
        parts.append(f"O PMI ficou em {float(pmi):.3f} (interação eletrostática/hidrofóbica estimada).")
    if interval and interval != "—":
        parts.append(f"O intervalo entre árvores do RF foi {interval}.")
    if q_peptide is not None:
        parts.append(f"Carga estimada do peptídeo: {float(q_peptide):.1f}.")
    if in_project:
        parts.append("Esta sequência já aparece no conjunto do projeto (Stigmurin/análogos).")
    if neighbors:
        top = neighbors[0]
        name = top.get("name") or top.get("peptide_id") or "vizinho"
        ident = top.get("identity")
        mic = top.get("mic_median_uM")
        extra = []
        if ident is not None:
            extra.append(f"identidade {100 * float(ident):.0f}%")
        if mic is not None:
            extra.append(f"MIC mediana {mic} µM")
        parts.append(
            "Vizinho mais próximo no treino: "
            + str(name)
            + (f" ({', '.join(extra)})" if extra else "")
            + "."
        )
    if shap_top:
        bits = []
        for row in shap_top[:3]:
            lab = row.get("label") or row.get("feature") or "?"
            val = row.get("shap_value")
            if val is None:
                continue
            sentido = "favorece" if float(val) >= 0 else "desfavorece"
            bits.append(f"{lab} ({sentido})")
        if bits:
            parts.append("No SHAP local, os principais fatores: " + "; ".join(bits) + ".")
    parts.append("Use isso para priorizar ensaios in vitro.")
    return " ".join(parts)


def template_explain_batch(
    *,
    target_label: str,
    rows: list[dict[str, Any]],
) -> str:
    """Resumo de lote a partir da tabela de predições (sem recalcular)."""
    ok = [r for r in rows if r.get("prob_calibrada") is not None]
    n = len(rows)
    n_ok = len(ok)
    n_high = sum(1 for r in ok if float(r["prob_calibrada"]) >= 0.70)
    n_mid = sum(1 for r in ok if 0.40 <= float(r["prob_calibrada"]) < 0.70)
    n_low = sum(1 for r in ok if float(r["prob_calibrada"]) < 0.40)
    parts = [
        f"Lote de {n} peptídeo(s) contra {target_label}: {n_ok} com predição válida; "
        f"{n_high} com ≥70% (fortes), {n_mid} intermediários e {n_low} com baixa probabilidade."
    ]
    ranked = sorted(ok, key=lambda r: float(r["prob_calibrada"]), reverse=True)
    if ranked:
        top = ranked[:3]
        bits = []
        for r in top:
            h = (r.get("header") or r.get("sequence") or "?")[:40]
            bits.append(f"{h} ({float(r['prob_calibrada']):.0%}, PMI {r.get('pmi')})")
        parts.append("Topo sugerido para priorizar: " + "; ".join(bits) + ".")
    parts.append("Ordenação e números vêm do Random Forest calibrado.")
    return " ".join(parts)


def template_explain_shap_overview(
    *,
    n_train: int,
    baseline_importance: list[dict[str, Any]] | None = None,
    multimodal_importance: list[dict[str, Any]] | None = None,
) -> str:
    """Narrativa do panorama SHAP global (barras + beeswarm)."""
    parts = [
        f"No treino atual ({n_train} pares MIC), o SHAP global mostra quais descritores "
        "mais influenciam a predição de alta atividade (MIC ≤ 3,4 µM) no Random Forest."
    ]

    def _top_labels(rows: list[dict[str, Any]] | None, k: int = 3) -> list[str]:
        if not rows:
            return []
        ranked = sorted(
            rows,
            key=lambda r: float(r.get("mean_abs_shap") or r.get("abs_shap") or abs(float(r.get("shap_value") or 0))),
            reverse=True,
        )
        out = []
        for r in ranked[:k]:
            lab = r.get("label") or r.get("feature") or "?"
            out.append(str(lab))
        return out

    base_top = _top_labels(baseline_importance)
    multi_top = _top_labels(multimodal_importance)
    if base_top:
        parts.append(
            "No baseline (clássicas + PMI), os fatores de maior |SHAP| médio são: "
            + ", ".join(base_top)
            + "."
        )
    if multi_top:
        parts.append(
            "No multimodal, destacam-se: "
            + ", ".join(multi_top)
            + "."
        )
    parts.append(
        "Barras/valores positivos empurram para alta atividade; negativos, para o contrário. "
        "No beeswarm, cada ponto é um par MIC: a cor indica o valor do descritor e a posição "
        "horizontal o impacto SHAP. Use isso para interpretar o modelo, não como prova biológica."
    )
    parts.append("Use isso para priorizar ensaios in vitro.")
    return " ".join(parts)


def narrate_shap_overview(
    *,
    n_train: int,
    baseline_importance: list[dict[str, Any]] | None = None,
    multimodal_importance: list[dict[str, Any]] | None = None,
    prefer_llm: bool = True,
) -> dict[str, str]:
    """Explica o SHAP global; GGUF opcional + fallback template."""
    tmpl = template_explain_shap_overview(
        n_train=n_train,
        baseline_importance=baseline_importance,
        multimodal_importance=multimodal_importance,
    )
    if not prefer_llm:
        return {"text": tmpl, "source": "template"}

    lines = [f"n_train={n_train}"]
    for tag, rows in (
        ("baseline", baseline_importance or []),
        ("multimodal", multimodal_importance or []),
    ):
        for i, r in enumerate(rows[:6]):
            lines.append(
                f"{tag}{i}={r.get('label')}:mean_abs={r.get('mean_abs_shap', r.get('shap_value'))}"
            )
    user = (
        "Explique o panorama SHAP global do PepMem-AI em português. "
        "Não invente números.\n\n"
        + "\n".join(lines)
        + "\n\nRascunho factual:\n"
        + tmpl
    )
    llm_text = _gguf_complete(user)
    if llm_text:
        return {"text": llm_text, "source": "qwen-gguf"}
    return {"text": tmpl, "source": "template"}


def _context_single(**kwargs: Any) -> str:
    """Serializa contexto factual para o LLM (só leitura)."""
    lines = [
        f"sequencia={kwargs.get('sequence')}",
        f"alvo={kwargs.get('target_label')}",
        f"prob_calibrada={kwargs.get('prob')}",
        f"pmi={kwargs.get('pmi')}",
        f"intervalo={kwargs.get('interval')}",
        f"q_peptide={kwargs.get('q_peptide')}",
        f"no_projeto={kwargs.get('in_project')}",
        f"faixa={_band_pt(float(kwargs.get('prob') or 0))}",
    ]
    neighbors = kwargs.get("neighbors") or []
    if neighbors:
        top = neighbors[0]
        lines.append(
            "vizinho0="
            f"{top.get('peptide_id')}/{top.get('name')};"
            f"identity={top.get('identity')};mic_med={top.get('mic_median_uM')}"
        )
    shap_top = kwargs.get("shap_top") or []
    for i, row in enumerate(shap_top[:5]):
        lines.append(
            f"shap{i}={row.get('label')}:{row.get('shap_value')}"
        )
    return "\n".join(lines)


def _gguf_complete(user_prompt: str, max_tokens: int = 280) -> str | None:
    llm, err = _get_llm()
    if llm is None:
        return None
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        text = out["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception as e:
        global _llm_error
        _llm_error = str(e)
        return None


def narrate_single(
    *,
    sequence: str,
    target_label: str,
    prob: float,
    pmi: float | None = None,
    interval: str | None = None,
    q_peptide: float | None = None,
    neighbors: list[dict[str, Any]] | None = None,
    shap_top: list[dict[str, Any]] | None = None,
    in_project: bool = False,
    prefer_llm: bool = True,
) -> dict[str, str]:
    """Gera narrativa; tenta GGUF e cai no template. Não altera predições."""
    tmpl = template_explain_single(
        sequence=sequence,
        target_label=target_label,
        prob=prob,
        pmi=pmi,
        interval=interval,
        q_peptide=q_peptide,
        neighbors=neighbors,
        shap_top=shap_top,
        in_project=in_project,
    )
    if not prefer_llm:
        return {"text": tmpl, "source": "template"}

    ctx = _context_single(
        sequence=sequence,
        target_label=target_label,
        prob=prob,
        pmi=pmi,
        interval=interval,
        q_peptide=q_peptide,
        neighbors=neighbors,
        shap_top=shap_top,
        in_project=in_project,
    )
    user = (
        "Explique estes resultados PepMem-AI para um pesquisador. "
        "Não invente valores.\n\nCONTEXTO:\n"
        f"{ctx}\n\n"
        "Se útil, você pode se basear neste rascunho factual:\n"
        f"{tmpl}"
    )
    llm_text = _gguf_complete(user)
    if llm_text:
        return {"text": llm_text, "source": "qwen-gguf"}
    return {"text": tmpl, "source": "template"}


def narrate_batch(
    *,
    target_label: str,
    rows: list[dict[str, Any]],
    prefer_llm: bool = True,
) -> dict[str, str]:
    """Narrativa de lote; GGUF opcional + fallback template."""
    tmpl = template_explain_batch(target_label=target_label, rows=rows)
    if not prefer_llm:
        return {"text": tmpl, "source": "template"}

    lines = [f"alvo={target_label}", f"n={len(rows)}"]
    for i, r in enumerate(rows[:12]):
        lines.append(
            f"row{i}: header={r.get('header')}; prob={r.get('prob_calibrada')}; "
            f"pmi={r.get('pmi')}; faixa={r.get('faixa')}"
        )
    user = (
        "Resuma este lote PepMem-AI em português (priorização in vitro). "
        "Não invente números.\n\n"
        + "\n".join(lines)
        + "\n\nRascunho factual:\n"
        + tmpl
    )
    llm_text = _gguf_complete(user, max_tokens=320)
    if llm_text:
        return {"text": llm_text, "source": "qwen-gguf"}
    return {"text": tmpl, "source": "template"}
