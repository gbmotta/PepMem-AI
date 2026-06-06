#!/usr/bin/env python3
"""Compute global SHAP importance for baseline and multimodal models."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepmem.shap_explain import plot_beeswarm, save_global_report


def main() -> None:
    import matplotlib.pyplot as plt

    models_dir = ROOT / "data" / "processed" / "models"

    baseline_path = models_dir / "baseline_mic_rf.joblib"
    if baseline_path.exists():
        pipe = joblib.load(baseline_path)
        report = save_global_report(pipe, use_embeddings=False, out_path="data/processed/models/shap_global_baseline.json")
        print("SHAP baseline:", report["n_samples"], "amostras")
        for row in report["global_importance"][:5]:
            print(f"  {row['label']}: {row['mean_abs_shap']:.4f}")
        fig = plot_beeswarm(pipe, False, title="Beeswarm SHAP — baseline")
        fig.savefig(models_dir / "shap_beeswarm_baseline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Beeswarm salvo:", models_dir / "shap_beeswarm_baseline.png")

    multi_path = models_dir / "multimodal_mic_rf.joblib"
    if multi_path.exists():
        pipe = joblib.load(multi_path)
        report = save_global_report(pipe, use_embeddings=True, out_path="data/processed/models/shap_global_multimodal.json")
        print("SHAP multimodal:", report["n_samples"], "amostras")
        for row in report["global_importance"][:5]:
            print(f"  {row['label']}: {row['mean_abs_shap']:.4f}")
        fig = plot_beeswarm(pipe, True, title="Beeswarm SHAP — multimodal")
        fig.savefig(models_dir / "shap_beeswarm_multimodal.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Beeswarm salvo:", models_dir / "shap_beeswarm_multimodal.png")


if __name__ == "__main__":
    main()
