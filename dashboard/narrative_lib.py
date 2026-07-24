"""Narrativa em português para o dashboard (templates, sem dependência frágil).

Fica em ``dashboard/`` para o Streamlit Cloud sempre achar o módulo ao lado de ``app.py``.
Tenta ``pepmem.narrative`` (Qwen GGUF) se disponível; senão usa só template.
"""

from __future__ import annotations

from typing import Any


def llm_status() -> dict[str, Any]:
    try:
        from pepmem.narrative import llm_status as _status

        return _status()
    except Exception as exc:  # noqa: BLE001
        return {
            "template_always": True,
            "llama_cpp_installed": False,
            "gguf_path": None,
            "gguf_ready": False,
            "auto_download": False,
            "last_error": str(exc),
        }


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
    parts: list[str] = []
    seq_short = sequence if len(sequence) <= 24 else f"{sequence[:21]}…"
    parts.append(
        f"Para o peptídeo {seq_short} contra {target_label}, o modelo calibrou "
        f"{prob:.0%} de chance de alta atividade (MIC ≤ 3,4 µM) — faixa: {_band_pt(prob)}."
    )
    if pmi is not None:
        parts.append(
            f"O PMI ficou em {float(pmi):.3f} (interação eletrostática/hidrofóbica estimada)."
        )
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


def template_explain_batch(*, target_label: str, rows: list[dict[str, Any]]) -> str:
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
        bits = []
        for r in ranked[:3]:
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
    parts = [
        f"No treino atual ({n_train} pares MIC), o SHAP global mostra quais descritores "
        "mais influenciam a predição de alta atividade (MIC ≤ 3,4 µM) no Random Forest."
    ]

    def _top_labels(rows: list[dict[str, Any]] | None, k: int = 3) -> list[str]:
        if not rows:
            return []
        ranked = sorted(
            rows,
            key=lambda r: float(
                r.get("mean_abs_shap") or r.get("abs_shap") or abs(float(r.get("shap_value") or 0))
            ),
            reverse=True,
        )
        return [str(r.get("label") or r.get("feature") or "?") for r in ranked[:k]]

    base_top = _top_labels(baseline_importance)
    multi_top = _top_labels(multimodal_importance)
    if base_top:
        parts.append(
            "No baseline (clássicas + PMI), os fatores de maior |SHAP| médio são: "
            + ", ".join(base_top)
            + "."
        )
    if multi_top:
        parts.append("No multimodal, destacam-se: " + ", ".join(multi_top) + ".")
    parts.append(
        "Barras/valores positivos empurram para alta atividade; negativos, para o contrário. "
        "No beeswarm, cada ponto é um par MIC: a cor indica o valor do descritor e a posição "
        "horizontal o impacto SHAP. Use isso para interpretar o modelo, não como prova biológica."
    )
    parts.append("Use isso para priorizar ensaios in vitro.")
    return " ".join(parts)


def template_explain_ranking(
    *,
    sequence: str,
    lambda_tox: float,
    rows: list[dict[str, Any]],
) -> str:
    """Narrativa do ranking multi-alvo."""
    seq_short = sequence if len(sequence) <= 24 else f"{sequence[:21]}…"
    n = len(rows)
    parts = [
        f"Ranking do peptídeo {seq_short} em {n} membrana(s), com penalização de "
        f"toxicidade λ = {lambda_tox:.2f}."
    ]
    ranked = sorted(
        [r for r in rows if r.get("final_score") is not None],
        key=lambda r: float(r["final_score"]),
        reverse=True,
    )
    if ranked:
        top = ranked[:3]
        bits = []
        for r in top:
            tid = r.get("target_id") or r.get("alvo") or "?"
            prob = r.get("pred_high_activity_prob")
            score = float(r["final_score"])
            pmi_sel = r.get("pmi_sel")
            bit = f"{tid} (score {score:.3f}"
            if prob is not None:
                try:
                    bit += f", prob {float(prob):.0%}"
                except (TypeError, ValueError):
                    bit += f", prob {prob}"
            if pmi_sel is not None:
                bit += f", PMI_sel {float(pmi_sel):.3f}"
            bit += ")"
            bits.append(bit)
        parts.append("Topo sugerido para priorizar ensaios: " + "; ".join(bits) + ".")
    parts.append(
        "O score final combina probabilidade de alta atividade, penalização de toxicidade "
        "(proxy em célula normal) e bônus de PMI seletivo. Ajuste λ se quiser mais ou menos "
        "cautela toxicológica."
    )
    parts.append("Use isso para priorizar ensaios in vitro.")
    return " ".join(parts)


def narrate_single(**kwargs: Any) -> dict[str, str]:
    try:
        from pepmem.narrative import narrate_single as _ns

        return _ns(**kwargs)
    except Exception:
        return {
            "text": template_explain_single(
                sequence=kwargs["sequence"],
                target_label=kwargs["target_label"],
                prob=float(kwargs["prob"]),
                pmi=kwargs.get("pmi"),
                interval=kwargs.get("interval"),
                q_peptide=kwargs.get("q_peptide"),
                neighbors=kwargs.get("neighbors"),
                shap_top=kwargs.get("shap_top"),
                in_project=bool(kwargs.get("in_project")),
            ),
            "source": "template",
        }


def narrate_batch(**kwargs: Any) -> dict[str, str]:
    try:
        from pepmem.narrative import narrate_batch as _nb

        return _nb(**kwargs)
    except Exception:
        return {
            "text": template_explain_batch(
                target_label=kwargs["target_label"],
                rows=kwargs["rows"],
            ),
            "source": "template",
        }


def narrate_shap_overview(**kwargs: Any) -> dict[str, str]:
    try:
        from pepmem.narrative import narrate_shap_overview as _no

        return _no(**kwargs)
    except Exception:
        return {
            "text": template_explain_shap_overview(
                n_train=int(kwargs["n_train"]),
                baseline_importance=kwargs.get("baseline_importance"),
                multimodal_importance=kwargs.get("multimodal_importance"),
            ),
            "source": "template",
        }


def narrate_ranking(**kwargs: Any) -> dict[str, str]:
    try:
        from pepmem.narrative import narrate_ranking as _nr

        return _nr(**kwargs)
    except Exception:
        return {
            "text": template_explain_ranking(
                sequence=kwargs["sequence"],
                lambda_tox=float(kwargs.get("lambda_tox") or 0.5),
                rows=kwargs["rows"],
            ),
            "source": "template",
        }
