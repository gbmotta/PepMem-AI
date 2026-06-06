#!/usr/bin/env python3
"""Train multimodal RF: classic features + ESM-2 embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pepmem"))
from features import CLASSIC_FEATURES, vectorize


def load_xy() -> tuple[np.ndarray, np.ndarray, list[str]]:
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    mic = pairs[pairs["mic_value"].notna()].copy()
    if mic.empty:
        raise SystemExit("Sem MIC para treino multimodal.")

    emb_path = ROOT / "data" / "processed" / "embeddings" / "esm2_all.npz"
    if not emb_path.exists():
        raise SystemExit("Execute generate_embeddings.py primeiro.")

    data = np.load(emb_path, allow_pickle=True)
    id_to_emb = dict(zip(data["peptide_ids"].tolist(), data["embeddings"]))

    X_list, y_list, ids = [], [], []
    for _, row in mic.iterrows():
        pid = row["peptide_id"]
        if pid not in id_to_emb:
            continue
        feats = row.to_dict()
        x = vectorize(feats, id_to_emb[pid], use_embeddings=True)
        X_list.append(x)
        y_list.append(int(row["mic_value"] <= 3.4))
        ids.append(pid)

    return np.vstack(X_list), np.array(y_list), ids


def main() -> None:
    X, y, _ = load_xy()
    n_features = X.shape[1]
    feature_names = CLASSIC_FEATURES + [f"esm2_{i}" for i in range(n_features - len(CLASSIC_FEATURES))]

    print(f"Amostras: {len(y)} | features: {n_features} | positivos: {y.sum()}/{len(y)}")

    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    preds = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", max_depth=6)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else None
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", max_depth=6)),
    ])
    final.fit(X, y)

    out = ROOT / "data" / "processed" / "models"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "multimodal_mic_rf.joblib")

    metrics = {
        "n_samples": len(y),
        "n_features": n_features,
        "feature_names": feature_names,
        "loo_auc": auc,
        "loo_classification_report": report,
        "label_rule": "MIC <= 3.4 uM => alta atividade",
        "model_type": "RandomForest + ESM-2 embeddings",
    }
    (out / "multimodal_mic_loo.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("LOO AUC multimodal:", auc)
    print("Modelo salvo:", out / "multimodal_mic_rf.joblib")


if __name__ == "__main__":
    main()
