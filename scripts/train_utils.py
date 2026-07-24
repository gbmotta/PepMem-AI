"""Utilitários compartilhados de treino: validação cruzada e calibração.

Fornece carregamento de pares MIC, pipeline Random Forest, avaliação LOO
(leave-one-out por amostra) e LOPO (leave-one-peptide-out), além de
calibração isotônica sobre probabilidades OOF do LOPO.

Rótulo binário: MIC ≤ 3,4 µM ⇒ alta atividade (``label_high_activity``).

Papel no pipeline: biblioteca usada por ``train_baseline.py`` e
``train_multimodal.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# --- features clássicas + PMI usadas pelo baseline ---
CLASSIC_FEATURES = [
    "q_peptide",
    "h_peptide",
    "mu_h_peptide",
    "surface_charge",
    "anionic_fraction",
    "cholesterol",
    "lps",
    "peptidoglycan",
    "ergosterol",
    "viral_envelope",
    "pmi",
]


def load_mic_pairs() -> pd.DataFrame:
    """Carrega pares com MIC e cria rótulo binário (MIC ≤ 3,4 µM = alta atividade)."""
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    mic = pairs[pairs["mic_value"].notna()].copy()
    if mic.empty:
        raise SystemExit("Sem pares MIC para treino.")
    mic["label_high_activity"] = (mic["mic_value"] <= 3.4).astype(int)
    return mic


def make_rf_pipeline(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Pipeline StandardScaler + RandomForest com ``class_weight='balanced'``."""
    clf_kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "class_weight": "balanced",
    }
    if max_depth is not None:
        clf_kwargs["max_depth"] = max_depth
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**clf_kwargs)),
        ]
    )


def evaluate_sample_loo(
    X: np.ndarray,
    y: np.ndarray,
    pipe_factory: Callable[[], Pipeline],
) -> dict[str, Any]:
    """LOO por amostra: cada linha (par peptídeo×alvo) é fold de teste.

    Métrica otimista — o mesmo peptídeo pode aparecer em treino e teste.
    Útil como referência, mas LOPO é a validação principal do projeto.
    """
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        pipe = pipe_factory()
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    return {
        "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None,
        "report": classification_report(y, preds, output_dict=True, zero_division=0),
        "probs": probs,
        "preds": preds,
    }


def evaluate_leave_one_peptide_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipe_factory: Callable[[], Pipeline],
) -> dict[str, Any]:
    """LOPO: deixa um peptídeo inteiro fora por fold (``LeaveOneGroupOut``).

    Evita vazamento entre alvos do mesmo peptídeo. As probabilidades OOF
    resultantes alimentam a calibração isotônica do modelo final.
    """
    logo = LeaveOneGroupOut()
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    n_groups = len(np.unique(groups))
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = pipe_factory()
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    # --- AUC global OOF + AUC por peptídeo (quando há ambas as classes) ---
    per_peptide: dict[str, float | None] = {}
    for g in np.unique(groups):
        mask = groups == g
        yy, pp = y[mask], probs[mask]
        if len(np.unique(yy)) > 1:
            per_peptide[str(g)] = float(roc_auc_score(yy, pp))
        else:
            per_peptide[str(g)] = None

    return {
        "n_peptides": int(n_groups),
        "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None,
        "report": classification_report(y, preds, output_dict=True, zero_division=0),
        "probs": probs,
        "preds": preds,
        "per_peptide_auc": per_peptide,
    }


def fit_isotonic_calibrator(oof_probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Calibração isotônica: mapeia probs OOF do LOPO para escala [0, 1].

    Ajusta uma regressão monotônica entre probabilidade bruta do RF e o
    rótulo observado, corrigindo desvio de calibração sem alterar ranking.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_probs, y.astype(float))
    return iso


def rf_tree_uncertainty(pipe: Pipeline, x: np.ndarray) -> tuple[float, float]:
    """Incerteza via dispersão das probabilidades entre árvores do RF."""
    scaler = pipe.named_steps["scaler"]
    clf: RandomForestClassifier = pipe.named_steps["clf"]
    xt = scaler.transform(x.reshape(1, -1))
    tree_probs = np.array([est.predict_proba(xt)[0, 1] for est in clf.estimators_])
    return float(tree_probs.mean()), float(tree_probs.std())
