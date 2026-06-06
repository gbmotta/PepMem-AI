#!/usr/bin/env python3
"""Train baseline classifiers on literature MIC endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
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


def load_training_data() -> pd.DataFrame:
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    mic = pairs[pairs["mic_value"].notna()].copy()
    if mic.empty:
        raise SystemExit("Sem dados MIC na literatura para treino.")
    mic["label_high_activity"] = (mic["mic_value"] <= 3.4).astype(int)
    return mic


def main() -> None:
    df = load_training_data()
    X = df[FEATURES].fillna(0).values
    y = df["label_high_activity"].values

    print(f"Amostras MIC: {len(df)} | alta atividade (MIC<=3.4): {y.sum()}/{len(y)}")

    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    report = classification_report(y, preds, output_dict=True, zero_division=0)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else None

    out_dir = ROOT / "data" / "processed" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modelo final em todos os dados
    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ]
    )
    final.fit(X, y)

    metrics = {
        "n_samples": len(y),
        "positive_rate": float(y.mean()),
        "loo_auc": auc,
        "loo_classification_report": report,
        "features": FEATURES,
        "label_rule": "MIC <= 3.4 uM => alta atividade",
    }
    (out_dir / "baseline_mic_loo.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import joblib

    joblib.dump(final, out_dir / "baseline_mic_rf.joblib")

    # Ranking dos pares do projeto sem endpoint
    all_pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    project_pairs = all_pairs.copy()
    X_all = project_pairs[FEATURES].fillna(0).values
    project_pairs["pred_high_activity_prob"] = final.predict_proba(X_all)[:, 1]
    project_pairs["pred_pmi_rank"] = project_pairs.groupby("target_id")["pmi"].rank(ascending=False)

    ranking = project_pairs.sort_values(["target_id", "pred_high_activity_prob"], ascending=[True, False])
    ranking.to_parquet(out_dir / "project_ranking_baseline.parquet", index=False)
    ranking.to_csv(out_dir / "project_ranking_baseline.csv", index=False)

    print("LOO AUC:", auc)
    print("Ranking salvo em", out_dir / "project_ranking_baseline.csv")


if __name__ == "__main__":
    main()
