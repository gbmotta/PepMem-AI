"""PepMem-AI prediction service."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from pepmem.features import CLASSIC_FEATURES, load_targets, pair_features, peptide_row_from_sequence, vectorize
from pepmem.paths import project_root

ROOT = project_root()
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"


class PepMemPredictor:
    def __init__(self, use_embeddings: bool = True) -> None:
        self.use_embeddings = use_embeddings
        self.targets = load_targets()
        self._embeddings_cache = self._load_embedding_index()
        self._model = self._load_model()
        self._esm = None
        self._tokenizer = None

    def _load_embedding_index(self) -> dict[str, np.ndarray]:
        path = ROOT / "data" / "processed" / "embeddings" / "esm2_all.npz"
        if not path.exists():
            return {}
        data = np.load(path, allow_pickle=True)
        ids = data["peptide_ids"].tolist()
        embs = data["embeddings"]
        by_seq: dict[str, np.ndarray] = {}
        base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")
        id_to_seq = base.set_index("peptide_id")["sequence"].to_dict()
        for pid, emb in zip(ids, embs):
            seq = id_to_seq.get(pid)
            if seq:
                by_seq[seq.upper()] = emb
        return by_seq

    def _load_model(self):
        models_dir = ROOT / "data" / "processed" / "models"
        name = "multimodal_mic_rf.joblib" if self.use_embeddings else "baseline_mic_rf.joblib"
        path = models_dir / name
        if not path.exists():
            path = models_dir / "baseline_mic_rf.joblib"
            self.use_embeddings = False
        return joblib.load(path)

    def _load_esm(self) -> None:
        if self._esm is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
        self._esm = AutoModel.from_pretrained(ESM_MODEL)
        self._esm.eval()

    def embed_sequence(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        if seq in self._embeddings_cache:
            return self._embeddings_cache[seq]
        self._load_esm()
        assert self._tokenizer and self._esm
        with torch.no_grad():
            inputs = self._tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
            out = self._esm(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb.squeeze().cpu().numpy()

    def predict_pair(
        self,
        sequence: str,
        target_id: str,
        net_charge: float | None = None,
    ) -> dict[str, Any]:
        x, _, feats = self._feature_vector(sequence, target_id, net_charge=net_charge)
        prob = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        return {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in {**feats, "pred_high_activity_prob": prob}.items()
        }

    def _feature_vector(
        self,
        sequence: str,
        target_id: str,
        net_charge: float | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        target = self.targets[self.targets["target_id"] == target_id]
        if target.empty:
            raise ValueError(f"Alvo desconhecido: {target_id}")
        peptide = peptide_row_from_sequence(sequence, net_charge=net_charge)
        feats = pair_features(peptide, target.iloc[0])
        emb = self.embed_sequence(sequence) if self.use_embeddings else None
        x = vectorize(feats, emb, self.use_embeddings)
        n_emb = len(emb) if emb is not None else 0
        from pepmem.shap_explain import feature_names

        names = feature_names(self.use_embeddings, n_emb or 320)
        return x, names, feats

    def explain_pair(
        self,
        sequence: str,
        target_id: str,
        net_charge: float | None = None,
    ) -> dict[str, Any]:
        from pepmem.shap_explain import explain_instance, load_training_matrix

        x, names, feats = self._feature_vector(sequence, target_id, net_charge=net_charge)
        try:
            X_bg, _, _ = load_training_matrix(self.use_embeddings)
            bg = self._model.named_steps["scaler"].transform(X_bg)
        except ValueError:
            bg = None

        explanation = explain_instance(self._model, x, names, background=bg)
        return {
            **feats,
            "pred_high_activity_prob": explanation["pred_high_activity_prob"],
            "expected_value_logit": explanation["expected_value_logit"],
            "shap_contributions": explanation["contributions"],
        }

    def global_shap_report(self) -> dict[str, Any] | None:
        fname = "shap_global_multimodal.json" if self.use_embeddings else "shap_global_baseline.json"
        path = ROOT / "data" / "processed" / "models" / fname
        if not path.exists():
            alt = ROOT / "data" / "processed" / "models" / "shap_global_baseline.json"
            path = alt if alt.exists() else path
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def rank_peptide(
        self,
        sequence: str,
        target_ids: list[str] | None = None,
        net_charge: float | None = None,
        lambda_tox: float = 0.5,
    ) -> pd.DataFrame:
        ids = target_ids or self.targets["target_id"].tolist()
        rows = [self.predict_pair(sequence, tid, net_charge=net_charge) for tid in ids]
        df = pd.DataFrame(rows)

        normal = df[df["target_id"] == "cell_normal"]
        pmi_normal = float(normal["pmi"].iloc[0]) if not normal.empty else 0.0
        df["pmi_normal"] = pmi_normal
        df["pmi_sel"] = df["pmi"] - pmi_normal

        tox = df[df["target_id"] == "cell_normal"]["pred_high_activity_prob"]
        tox_score = float(tox.iloc[0]) if not tox.empty else 0.0
        df["toxicity_proxy"] = tox_score
        df["final_score"] = df["pred_high_activity_prob"] - lambda_tox * tox_score
        df.loc[df["target_id"] == "cell_normal", "final_score"] = np.nan
        df["pmi_sel_bonus"] = df["pmi_sel"].clip(lower=0) * 0.1
        df["final_score"] = df["final_score"] + df["pmi_sel_bonus"]

        df = df.sort_values("final_score", ascending=False, na_position="last")
        df = df.replace({np.nan: None})
        return df

    def list_targets(self) -> list[dict[str, Any]]:
        return self.targets[
            ["target_id", "target", "target_type", "surface_charge", "anionic_fraction"]
        ].to_dict(orient="records")

    @property
    def model_info(self) -> dict[str, Any]:
        meta_path = ROOT / "data" / "processed" / "models" / "multimodal_mic_loo.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        meta_path = ROOT / "data" / "processed" / "models" / "baseline_mic_loo.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}
