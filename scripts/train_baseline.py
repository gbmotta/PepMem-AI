#!/usr/bin/env python3
"""Train baseline RF on MIC endpoints with leave-one-peptide-out + calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_utils import (
    CLASSIC_FEATURES,
    evaluate_leave_one_peptide_out,
    evaluate_sample_loo,
    fit_isotonic_calibrator,
    load_mic_pairs,
    make_rf_pipeline,
)


def main() -> None:
    df = load_mic_pairs()
    X = df[CLASSIC_FEATURES].fillna(0).values
    y = df["label_high_activity"].values
    groups = df["peptide_id"].astype(str).values

    print(f"Amostras MIC: {len(df)} | peptídeos: {len(np.unique(groups))} | alta atividade: {y.sum()}/{len(y)}")

    factory = lambda: make_rf_pipeline(n_estimators=200)
    sample = evaluate_sample_loo(X, y, factory)
    peptide = evaluate_leave_one_peptide_out(X, y, groups, factory)

    print(f"LOO amostra AUC: {sample['auc']:.4f}" if sample["auc"] else "LOO amostra AUC: n/a")
    print(f"Leave-one-peptide AUC: {peptide['auc']:.4f}" if peptide["auc"] else "Leave-one-peptide AUC: n/a")

    calibrator = fit_isotonic_calibrator(peptide["probs"], y)
    cal_probs = calibrator.predict(peptide["probs"])

    out_dir = ROOT / "data" / "processed" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    final = make_rf_pipeline(n_estimators=200)
    final.fit(X, y)
    joblib.dump(final, out_dir / "baseline_mic_rf.joblib")
    joblib.dump(calibrator, out_dir / "baseline_mic_calibrator.joblib")

    metrics = {
        "n_samples": len(y),
        "n_peptides": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "loo_auc": sample["auc"],
        "leave_one_peptide_auc": peptide["auc"],
        "per_peptide_auc": peptide["per_peptide_auc"],
        "loo_classification_report": sample["report"],
        "leave_one_peptide_classification_report": peptide["report"],
        "features": CLASSIC_FEATURES,
        "label_rule": "MIC <= 3.4 uM => alta atividade",
        "calibration": "isotonic_on_leave_one_peptide_oof",
        "model_type": "RandomForest baseline (clássicas + PMI)",
    }
    (out_dir / "baseline_mic_loo.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    oof = df[["peptide_id", "target_id", "mic_value", "label_high_activity"]].copy()
    oof["prob_raw_lope"] = peptide["probs"]
    oof["prob_calibrated_lope"] = cal_probs
    oof.to_csv(out_dir / "baseline_oof_probs.csv", index=False)

    all_pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    project_pairs = all_pairs.copy()
    X_all = project_pairs[CLASSIC_FEATURES].fillna(0).values
    raw = final.predict_proba(X_all)[:, 1]
    project_pairs["pred_high_activity_prob"] = calibrator.predict(raw)
    project_pairs["pred_high_activity_prob_raw"] = raw
    project_pairs["pred_pmi_rank"] = project_pairs.groupby("target_id")["pmi"].rank(ascending=False)
    ranking = project_pairs.sort_values(
        ["target_id", "pred_high_activity_prob"], ascending=[True, False]
    )
    ranking.to_parquet(out_dir / "project_ranking_baseline.parquet", index=False)
    ranking.to_csv(out_dir / "project_ranking_baseline.csv", index=False)

    print("Calibrador isotônico salvo.")
    print("Ranking salvo em", out_dir / "project_ranking_baseline.csv")


if __name__ == "__main__":
    main()
