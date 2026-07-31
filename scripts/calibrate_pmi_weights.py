"""Calibra pesos α, β, γ, δ do PMI nos pares com MIC (leave-peptide-out).

Aviso: com ~10 peptídeos / 90 pares o risco de overfitting é alto. Os pesos
calibrados são ótimos *neste* dataset sob LOPO, não universais.

Uso:
  PYTHONPATH=. python scripts/calibrate_pmi_weights.py
  PYTHONPATH=. python scripts/calibrate_pmi_weights.py --apply-defaults

Saídas em ``data/processed/models/``:
  - pmi_weights_calibration.json
  - pmi_weights_calibrated.json  (pesos recomendados + métricas)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pepmem.peptide_utils import compute_mean_hydrophobicity, compute_net_charge  # noqa: E402
from pepmem.pmi import (  # noqa: E402
    DEFAULT_WEIGHTS,
    hydrophobic_moment,
    membrane_h_proxy,
    sterol_penalty,
)
from train_utils import CLASSIC_FEATURES, load_mic_pairs, make_rf_pipeline  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models"


def _prepare_components(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche q/h/μH e h_m/esterol a partir da sequência/alvo quando faltam."""
    out = df.copy()
    qs, hs, mus, hms, sterols = [], [], [], [], []
    for _, row in out.iterrows():
        seq = str(row.get("sequence") or "")
        q = row.get("q_peptide")
        if q is None or (isinstance(q, float) and np.isnan(q)):
            q = compute_net_charge(seq) or 0.0
        h = row.get("h_peptide")
        if h is None or (isinstance(h, float) and np.isnan(h)):
            h = compute_mean_hydrophobicity(seq) or 0.0
        mu = row.get("mu_h_peptide")
        if mu is None or (isinstance(mu, float) and np.isnan(mu)):
            mu = hydrophobic_moment(seq) if seq else 0.0
        hm = row.get("h_membrane")
        if hm is None or (isinstance(hm, float) and np.isnan(hm)):
            hm = membrane_h_proxy(row.get("target_type"))
        chol = row.get("cholesterol")
        chol = 0.0 if chol is None or (isinstance(chol, float) and np.isnan(chol)) else float(chol)
        ergo = row.get("ergosterol")
        ergo = 0.0 if ergo is None or (isinstance(ergo, float) and np.isnan(ergo)) else float(ergo)
        st = row.get("sterol_penalty")
        if st is None or (isinstance(st, float) and np.isnan(st)):
            st = sterol_penalty(chol, ergo)
        qs.append(float(q))
        hs.append(float(h))
        mus.append(float(mu))
        hms.append(float(hm))
        sterols.append(float(st))
    out["q_peptide"] = qs
    out["h_peptide"] = hs
    out["mu_h_peptide"] = mus
    out["h_membrane"] = hms
    out["sterol_penalty"] = sterols
    out["surface_charge"] = out["surface_charge"].astype(float).fillna(0.0)
    out["cholesterol"] = out["cholesterol"].astype(float).fillna(0.0)
    return out


def compute_pmi_vector(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> np.ndarray:
    """PMI = α q|q_m| + β h h_m + γ μH − δ esterol."""
    return (
        alpha * df["q_peptide"].to_numpy() * np.abs(df["surface_charge"].to_numpy())
        + beta * df["h_peptide"].to_numpy() * df["h_membrane"].to_numpy()
        + gamma * df["mu_h_peptide"].to_numpy()
        - delta * df["sterol_penalty"].to_numpy()
    )


def auc_safe(y: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def weight_grid() -> list[dict[str, float]]:
    """Grade compacta (~256 combinações) em torno dos defaults."""
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    gammas = [0.0, 0.15, 0.3, 0.45, 0.6]
    deltas = [0.0, 0.2, 0.4, 0.6, 0.8]
    return [
        {"alpha": a, "beta": b, "gamma": g, "delta": d}
        for a, b, g, d in itertools.product(alphas, betas, gammas, deltas)
    ]


def best_weights_on_subset(df: pd.DataFrame, grid: list[dict[str, float]]) -> tuple[dict[str, float], float | None]:
    """Escolhe pesos que maximizam AUC(PMI, y) no subconjunto."""
    y = df["label_high_activity"].to_numpy()
    best_w = dict(DEFAULT_WEIGHTS)
    best_auc: float | None = None
    for w in grid:
        scores = compute_pmi_vector(df, **w)
        auc = auc_safe(y, scores)
        if auc is None:
            continue
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_w = w
    return best_w, best_auc


def nested_lopo_pmi_auc(df: pd.DataFrame, grid: list[dict[str, float]]) -> dict:
    """Para cada peptídeo fora: calibra pesos no resto e avalia no held-out."""
    groups = df["peptide_id"].astype(str).to_numpy()
    y = df["label_high_activity"].to_numpy()
    logo = LeaveOneGroupOut()
    oof = np.full(len(df), np.nan)
    fold_rows = []
    for train_idx, test_idx in logo.split(df, y, groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        w, train_auc = best_weights_on_subset(train_df, grid)
        scores = compute_pmi_vector(test_df, **w)
        oof[test_idx] = scores
        fold_rows.append(
            {
                "peptide_id": str(test_df["peptide_id"].iloc[0]),
                "n_test": int(len(test_idx)),
                "train_auc_pmi": train_auc,
                "test_auc_pmi": auc_safe(
                    test_df["label_high_activity"].to_numpy(), scores
                ),
                **w,
            }
        )
    return {
        "oof_auc_pmi": auc_safe(y, oof),
        "folds": fold_rows,
        "oof_scores": oof.tolist(),
    }


def rf_lopo_auc(df: pd.DataFrame, pmi: np.ndarray) -> float | None:
    """LOPO do RF baseline clássico com coluna PMI substituída."""
    work = df.copy()
    work["pmi"] = pmi
    X = work[CLASSIC_FEATURES].fillna(0).to_numpy()
    y = work["label_high_activity"].to_numpy()
    groups = work["peptide_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y))
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = make_rf_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    return auc_safe(y, probs)


def apply_defaults(weights: dict[str, float]) -> None:
    """Reescreve DEFAULT_WEIGHTS em pepmem/pmi.py (e espelho scripts/pmi.py se existir)."""
    paths = [ROOT / "pepmem" / "pmi.py", ROOT / "scripts" / "pmi.py"]
    old = (
        'DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 0.3, "delta": 0.4}'
    )
    new = (
        "DEFAULT_WEIGHTS = {"
        f'"alpha": {weights["alpha"]}, '
        f'"beta": {weights["beta"]}, '
        f'"gamma": {weights["gamma"]}, '
        f'"delta": {weights["delta"]}'
        "}"
    )
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "DEFAULT_WEIGHTS = {" not in text:
            continue
        # substitui a linha DEFAULT_WEIGHTS = {...}
        lines = []
        for line in text.splitlines(keepends=True):
            if line.strip().startswith("DEFAULT_WEIGHTS"):
                nl = "\n" if line.endswith("\n") and not new.endswith("\n") else ""
                lines.append(f"{new}{nl}" if not new.endswith("\n") else new + ("\n" if line.endswith("\n") else ""))
            else:
                lines.append(line)
        path.write_text("".join(lines), encoding="utf-8")
        print(f"Atualizado DEFAULT_WEIGHTS em {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply-defaults",
        action="store_true",
        help="Grava os pesos calibrados (fit em todos os 90) em pepmem/pmi.py",
    )
    args = parser.parse_args()

    df = _prepare_components(load_mic_pairs())
    grid = weight_grid()
    y = df["label_high_activity"].to_numpy()

    # --- baselines com pesos atuais ---
    default_pmi = compute_pmi_vector(df, **DEFAULT_WEIGHTS)
    default_auc = auc_safe(y, default_pmi)
    default_rf = rf_lopo_auc(df, default_pmi)

    # --- LOPO aninhado (estimativa mais honesta) ---
    nested = nested_lopo_pmi_auc(df, grid)

    # --- pesos "recomendados": fit em todos (otimista; para uso + documentação) ---
    fitted, fitted_auc = best_weights_on_subset(df, grid)
    fitted_pmi = compute_pmi_vector(df, **fitted)
    fitted_rf = rf_lopo_auc(df, fitted_pmi)

    report = {
        "n_pairs": int(len(df)),
        "n_peptides": int(df["peptide_id"].nunique()),
        "n_grid": len(grid),
        "h_membrane_note": "proxy por target_type / membrane_hydrophobicity (não mais 0.5 fixo)",
        "sterol_note": "sterol_penalty = cholesterol + 0.8*ergosterol",
        "warning": (
            "Pesos calibrados em dataset pequeno (~10 peptídeos). "
            "LOPO aninhado reduz vazamento, mas não elimina overfitting."
        ),
        "default_weights": dict(DEFAULT_WEIGHTS),
        "default_auc_pmi_full": default_auc,
        "default_auc_rf_lopo": default_rf,
        "nested_lopo_auc_pmi": nested["oof_auc_pmi"],
        "nested_folds": nested["folds"],
        "calibrated_weights_full_fit": fitted,
        "calibrated_auc_pmi_full": fitted_auc,
        "calibrated_auc_rf_lopo": fitted_rf,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUT_DIR / "pmi_weights_calibration.json"
    full_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    slim = {
        "alpha": fitted["alpha"],
        "beta": fitted["beta"],
        "gamma": fitted["gamma"],
        "delta": fitted["delta"],
        "source": "calibrate_pmi_weights.py full-fit on 90 MIC pairs",
        "nested_lopo_auc_pmi": nested["oof_auc_pmi"],
        "default_auc_pmi_full": default_auc,
        "calibrated_auc_pmi_full": fitted_auc,
        "default_auc_rf_lopo": default_rf,
        "calibrated_auc_rf_lopo": fitted_rf,
        "warning": report["warning"],
    }
    slim_path = OUT_DIR / "pmi_weights_calibrated.json"
    slim_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Calibração PMI (90 pares, LOPO) ===")
    print(f"Default {DEFAULT_WEIGHTS}")
    print(f"  AUC(PMI) full:     {default_auc}")
    print(f"  AUC(RF) LOPO:      {default_rf}")
    print(f"Nested LOPO AUC(PMI): {nested['oof_auc_pmi']}")
    print(f"Calibrado (full-fit): {fitted}")
    print(f"  AUC(PMI) full:     {fitted_auc}")
    print(f"  AUC(RF) LOPO:      {fitted_rf}")
    print(f"Relatório: {full_path}")
    print(f"Pesos:     {slim_path}")

    if args.apply_defaults:
        apply_defaults(fitted)
        print(
            "Defaults atualizados. Rode build_pairs + train_baseline para "
            "propagar o PMI novo aos pares/modelo."
        )


if __name__ == "__main__":
    main()
