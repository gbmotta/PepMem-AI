#!/usr/bin/env python3
"""Train multimodal RF (clássicas + ESM-2) with leave-one-peptide-out + calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pepmem"))

from features import CLASSIC_FEATURES, vectorize
from train_utils import (
    evaluate_leave_one_peptide_out,
    evaluate_sample_loo,
    fit_isotonic_calibrator,
    load_mic_pairs,
    make_rf_pipeline,
)


def load_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    mic = load_mic_pairs()
    emb_path = ROOT / "data" / "processed" / "embeddings" / "esm2_all.npz"
    if not emb_path.exists():
        raise SystemExit("Execute generate_embeddings.py primeiro.")

    data = np.load(emb_path, allow_pickle=True)
    id_to_emb = dict(zip(data["peptide_ids"].tolist(), data["embeddings"]))

    X_list, y_list, groups, kept = [], [], [], []
    for _, row in mic.iterrows():
        pid = row["peptide_id"]
        if pid not in id_to_emb:
            continue
        X_list.append(vectorize(row.to_dict(), id_to_emb[pid], use_embeddings=True))
        y_list.append(int(row["label_high_activity"]))
        groups.append(str(pid))
        kept.append(
            {
                "peptide_id": pid,
                "target_id": row["target_id"],
                "mic_value": row["mic_value"],
                "label_high_activity": int(row["label_high_activity"]),
            }
        )
    if not X_list:
        raise SystemExit("Nenhuma amostra com embedding.")
    return np.vstack(X_list), np.array(y_list), np.array(groups), kept


def main() -> None:
    X, y, groups, kept_rows = load_xy()
    n_features = X.shape[1]
    feature_names = CLASSIC_FEATURES + [f"esm2_{i}" for i in range(n_features - len(CLASSIC_FEATURES))]

    print(f"Amostras: {len(y)} | peptídeos: {len(np.unique(groups))} | features: {n_features} | positivos: {y.sum()}/{len(y)}")

    factory = lambda: make_rf_pipeline(n_estimators=300, max_depth=6)
    sample = evaluate_sample_loo(X, y, factory)
    peptide = evaluate_leave_one_peptide_out(X, y, groups, factory)

    print(f"LOO amostra AUC: {sample['auc']}")
    print(f"Leave-one-peptide AUC: {peptide['auc']}")

    calibrator = fit_isotonic_calibrator(peptide["probs"], y)

    final = make_rf_pipeline(n_estimators=300, max_depth=6)
    final.fit(X, y)

    out = ROOT / "data" / "processed" / "models"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "multimodal_mic_rf.joblib")
    joblib.dump(calibrator, out / "multimodal_mic_calibrator.joblib")

    metrics = {
        "n_samples": len(y),
        "n_peptides": int(len(np.unique(groups))),
        "n_features": n_features,
        "feature_names": feature_names,
        "loo_auc": sample["auc"],
        "leave_one_peptide_auc": peptide["auc"],
        "per_peptide_auc": peptide["per_peptide_auc"],
        "loo_classification_report": sample["report"],
        "leave_one_peptide_classification_report": peptide["report"],
        "label_rule": "MIC <= 3.4 uM => alta atividade",
        "calibration": "isotonic_on_leave_one_peptide_oof",
        "model_type": "RandomForest + ESM-2 embeddings",
    }
    (out / "multimodal_mic_loo.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    oof = pd.DataFrame(kept_rows)
    oof["prob_raw_lope"] = peptide["probs"]
    oof["prob_calibrated_lope"] = calibrator.predict(peptide["probs"])
    oof.to_csv(out / "multimodal_oof_probs.csv", index=False)

    print("LOO AUC multimodal (amostra):", sample["auc"])
    print("Leave-one-peptide AUC multimodal:", peptide["auc"])
    print("Modelo + calibrador salvos em", out)


if __name__ == "__main__":
    main()
