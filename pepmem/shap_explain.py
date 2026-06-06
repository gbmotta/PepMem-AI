"""SHAP explanations for PepMem-AI Random Forest models."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pepmem.features import CLASSIC_FEATURES, vectorize
from pepmem.paths import project_root

ROOT = project_root()

FEATURE_LABELS: dict[str, str] = {
    "q_peptide": "Carga peptídeo (q)",
    "h_peptide": "Hidrofobicidade peptídeo",
    "mu_h_peptide": "Momento hidrofóbico",
    "surface_charge": "Carga superficial membrana",
    "anionic_fraction": "Fração aniónica membrana",
    "cholesterol": "Colesterol",
    "lps": "LPS (Gram−)",
    "peptidoglycan": "Peptidoglicano (Gram+)",
    "ergosterol": "Ergosterol (fungo)",
    "viral_envelope": "Envelope viral",
    "pmi": "PMI",
    "esm2_embedding": "ESM-2 (embedding agregado)",
}


def feature_names(use_embeddings: bool, n_embedding_dims: int = 320) -> list[str]:
    if not use_embeddings:
        return list(CLASSIC_FEATURES)
    return list(CLASSIC_FEATURES) + [f"esm2_{i}" for i in range(n_embedding_dims)]


def _label(name: str) -> str:
    if name.startswith("esm2_"):
        return FEATURE_LABELS["esm2_embedding"]
    return FEATURE_LABELS.get(name, name)


def load_training_matrix(use_embeddings: bool) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load MIC-labelled pairs as feature matrix for SHAP background."""
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    mic = pairs[pairs["mic_value"].notna()].copy()
    if mic.empty:
        raise ValueError("Sem dados MIC para explicação SHAP.")

    rows: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []

    if use_embeddings:
        emb_path = ROOT / "data" / "processed" / "embeddings" / "esm2_all.npz"
        if not emb_path.exists():
            raise ValueError("Embeddings ESM-2 ausentes. Execute generate_embeddings.py.")
        data = np.load(emb_path, allow_pickle=True)
        id_to_emb = dict(zip(data["peptide_ids"].tolist(), data["embeddings"]))
        n_emb = int(data["embeddings"].shape[1])
        names = feature_names(True, n_emb)
        for _, row in mic.iterrows():
            pid = row["peptide_id"]
            if pid not in id_to_emb:
                continue
            x = vectorize(row.to_dict(), id_to_emb[pid], use_embeddings=True)
            rows.append(x)
            kept.append(row.to_dict())
    else:
        names = feature_names(False)
        for _, row in mic.iterrows():
            x = vectorize(row.to_dict(), None, use_embeddings=False)
            rows.append(x)
            kept.append(row.to_dict())

    if not rows:
        raise ValueError("Nenhuma amostra MIC com features disponíveis.")

    return np.vstack(rows), names, pd.DataFrame(kept)


def _positive_class_shap(shap_values: Any, sample_idx: int = 0) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[sample_idx, :, 1]
    if arr.ndim == 2:
        return arr[sample_idx]
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])[sample_idx]
    raise ValueError("Formato de SHAP values não reconhecido.")


def _aggregate_shap(shap_row: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    classic: list[dict[str, Any]] = []
    esm_total = 0.0
    has_esm = False

    for name, value in zip(names, shap_row):
        val = float(value)
        if name.startswith("esm2_"):
            esm_total += val
            has_esm = True
        else:
            classic.append(
                {
                    "feature": name,
                    "label": _label(name),
                    "shap_value": val,
                    "abs_shap": abs(val),
                    "group": "classic",
                }
            )

    if has_esm:
        classic.append(
            {
                "feature": "esm2_embedding",
                "label": FEATURE_LABELS["esm2_embedding"],
                "shap_value": esm_total,
                "abs_shap": abs(esm_total),
                "group": "embedding",
            }
        )

    classic.sort(key=lambda r: r["abs_shap"], reverse=True)
    return classic


def _all_positive_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _aggregate_matrix_for_display(
    X: np.ndarray,
    shap_matrix: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classic_idx = [i for i, n in enumerate(names) if not n.startswith("esm2_")]
    esm_idx = [i for i, n in enumerate(names) if n.startswith("esm2_")]

    if not esm_idx:
        return X, shap_matrix, [_label(n) for n in names]

    x_classic = X[:, classic_idx]
    sv_classic = shap_matrix[:, classic_idx]
    x_esm = np.linalg.norm(X[:, esm_idx], axis=1, keepdims=True)
    sv_esm = shap_matrix[:, esm_idx].sum(axis=1, keepdims=True)
    labels = [_label(names[i]) for i in classic_idx] + [FEATURE_LABELS["esm2_embedding"]]
    return np.hstack([x_classic, x_esm]), np.hstack([sv_classic, sv_esm]), labels


def compute_training_shap(
    pipeline: Pipeline,
    use_embeddings: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    X, names, meta = load_training_matrix(use_embeddings)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    x_scaled = scaler.transform(X)

    import shap

    explainer = shap.TreeExplainer(clf, data=x_scaled)
    shap_values = explainer.shap_values(x_scaled)
    sv = _all_positive_shap(shap_values)
    return *_aggregate_matrix_for_display(X, sv, names), meta


def plot_beeswarm(
    pipeline: Pipeline,
    use_embeddings: bool,
    title: str = "SHAP beeswarm — amostras MIC de treino",
) -> Any:
    """Classic SHAP summary beeswarm (dot plot) on training MIC pairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    x_display, sv, labels, _ = compute_training_shap(pipeline, use_embeddings)
    order = np.argsort(np.mean(np.abs(sv), axis=0))[::-1]
    x_display = x_display[:, order]
    sv = sv[:, order]
    labels = [labels[i] for i in order]

    fig_height = max(5.0, len(labels) * 0.55)
    plt.figure(figsize=(11, fig_height))
    shap.summary_plot(
        sv,
        x_display,
        feature_names=labels,
        plot_type="dot",
        show=False,
        color_bar_label="Valor do descritor",
        max_display=len(labels),
    )
    fig = plt.gcf()
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def explain_instance(
    pipeline: Pipeline,
    x: np.ndarray,
    feature_names_list: list[str],
    background: np.ndarray | None = None,
) -> dict[str, Any]:
    """SHAP TreeExplainer for one scaled feature vector."""
    import shap

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    bg = background if background is not None else scaler.transform(x.reshape(1, -1))
    x_scaled = scaler.transform(x.reshape(1, -1))

    explainer = shap.TreeExplainer(clf, data=bg)
    shap_values = explainer.shap_values(x_scaled)
    shap_row = _positive_class_shap(shap_values, 0)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])
    else:
        base_value = float(base_value)

    contributions = _aggregate_shap(shap_row, feature_names_list)
    prob = float(clf.predict_proba(x_scaled)[0, 1])

    return {
        "expected_value_logit": base_value,
        "pred_high_activity_prob": prob,
        "contributions": contributions,
        "feature_names": feature_names_list,
    }


def global_importance(
    pipeline: Pipeline,
    X: np.ndarray,
    feature_names_list: list[str],
) -> list[dict[str, Any]]:
    """Mean |SHAP| across training samples (global importance)."""
    import shap

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(clf, data=X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs = np.mean(np.abs(sv), axis=0)
    rows = _aggregate_shap(mean_abs, feature_names_list)
    for row in rows:
        row["mean_abs_shap"] = row.pop("abs_shap")
        row["shap_value"] = row["mean_abs_shap"]
    return rows


def save_global_report(
    pipeline: Pipeline,
    use_embeddings: bool,
    out_path: str,
) -> dict[str, Any]:
    X, names, meta = load_training_matrix(use_embeddings)
    importance = global_importance(pipeline, X, names)
    report = {
        "model": "multimodal_mic_rf" if use_embeddings else "baseline_mic_rf",
        "n_samples": len(X),
        "n_features_raw": len(names),
        "global_importance": importance,
        "training_pairs": meta[["peptide_id", "target_id", "mic_value"]].to_dict(orient="records"),
    }
    path = ROOT / out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def plot_contributions(contributions: list[dict[str, Any]], title: str = "SHAP"):
    """Horizontal bar chart of SHAP contributions (matplotlib figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c["label"] for c in reversed(contributions)]
    values = [c["shap_value"] for c in reversed(contributions)]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]

    height = max(3.5, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Contribuição SHAP → alta atividade")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, values):
        x = bar.get_width()
        ax.text(
            x + (0.01 if x >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def plot_global_importance(importance: list[dict[str, Any]], title: str = "Importância global |SHAP|"):
    """Bar chart of mean |SHAP| across training MIC samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(importance, key=lambda r: r["mean_abs_shap"])
    labels = [r["label"] for r in sorted_rows]
    values = [r["mean_abs_shap"] for r in sorted_rows]
    groups = [r.get("group", "classic") for r in sorted_rows]
    colors = ["#1f77b4" if g == "classic" else "#9467bd" for g in groups]

    height = max(3.5, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Média |SHAP| (12 MICs de treino)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    fig.tight_layout()
    return fig
