"""Serviço de predição PepMem-AI (inferência + vizinhos + ranking).

Carrega o Random Forest (multimodal ou baseline), calibrador isotônico OOF
e índice de MICs. Expõe:

- ``predict_pair`` — probabilidade calibrada de alta atividade + incerteza
- ``find_neighbors`` — peptídeos do treino por identidade / cosine ESM
- ``explain_pair`` — contribuições SHAP locais
- ``rank_peptide`` — ranking de alvos com penalidade de toxicidade proxy

Papel no pipeline: API usada pelo dashboard Streamlit e por clientes Python.
"""

from __future__ import annotations

import json
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


def _seq_identity(a: str, b: str) -> float:
    """Identidade de sequência por posições alinhadas ao início (≈ global curta)."""
    a, b = a.upper(), b.upper()
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    m = min(len(a), len(b))
    same = sum(x == y for x, y in zip(a[:m], b[:m]))
    return same / n


class PepMemPredictor:
    """Orquestra modelo, embeddings e índice MIC para uma sessão de inferência."""

    def __init__(self, use_embeddings: bool = True) -> None:
        self.use_embeddings = use_embeddings
        self.targets = load_targets()
        self._embeddings_cache = self._load_embedding_index()
        self._model = self._load_model()
        self._calibrator = self._load_calibrator()
        self._mic_index = self._load_mic_index()
        # ESM carregado sob demanda (pesado) só se a sequência não estiver no cache
        self._esm = None
        self._tokenizer = None

    def _load_embedding_index(self) -> dict[str, np.ndarray]:
        """Índice sequência → embedding pré-computado (``esm2_all.npz``)."""
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
                by_seq[str(seq).upper()] = emb
        return by_seq

    def _load_model(self):
        """Carrega RF multimodal; cai para baseline se o multimodal não existir."""
        models_dir = ROOT / "data" / "processed" / "models"
        name = "multimodal_mic_rf.joblib" if self.use_embeddings else "baseline_mic_rf.joblib"
        path = models_dir / name
        if not path.exists():
            path = models_dir / "baseline_mic_rf.joblib"
            self.use_embeddings = False
        return joblib.load(path)

    def _load_calibrator(self):
        """IsotonicRegression ajustado em probs OOF leave-one-peptide-out (opcional)."""
        models_dir = ROOT / "data" / "processed" / "models"
        name = (
            "multimodal_mic_calibrator.joblib"
            if self.use_embeddings
            else "baseline_mic_calibrator.joblib"
        )
        path = models_dir / name
        if not path.exists():
            alt = models_dir / "baseline_mic_calibrator.joblib"
            path = alt if alt.exists() else path
        if not path.exists():
            return None
        return joblib.load(path)

    def _load_mic_index(self) -> pd.DataFrame:
        """Pares com MIC conhecido — base dos vizinhos kNN no dashboard."""
        path = ROOT / "data" / "processed" / "pepmem_pairs.parquet"
        if not path.exists():
            return pd.DataFrame()
        pairs = pd.read_parquet(path)
        mic = pairs[pairs["mic_value"].notna()].copy()
        if mic.empty:
            return mic
        proj = ROOT / "data" / "processed" / "pepmem_base_project.csv"
        names = {}
        if proj.exists():
            pdf = pd.read_csv(proj)
            names = pdf.set_index("peptide_id")["name"].to_dict()
        mic["name"] = mic["peptide_id"].map(names)
        return mic

    def _load_esm(self) -> None:
        """Lazy-load do ESM-2 pequeno (t6_8M) para sequências novas."""
        if self._esm is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
        self._esm = AutoModel.from_pretrained(ESM_MODEL)
        self._esm.eval()

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Retorna embedding mean-pooled; usa cache em disco quando a seq já existe."""
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
        vec = emb.squeeze().cpu().numpy()
        self._embeddings_cache[seq] = vec
        return vec

    def _raw_and_calibrated_prob(self, x: np.ndarray) -> tuple[float, float, float]:
        """Probabilidade bruta do RF, calibrada (isotonic) e σ entre árvores."""
        raw = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        # Discordância entre árvores ≈ incerteza epistêmica do ensemble
        scaler = self._model.named_steps["scaler"]
        clf = self._model.named_steps["clf"]
        xt = scaler.transform(x.reshape(1, -1))
        tree_probs = np.array([est.predict_proba(xt)[0, 1] for est in clf.estimators_])
        std = float(tree_probs.std())
        if self._calibrator is not None:
            cal = float(self._calibrator.predict([raw])[0])
        else:
            cal = raw
        return raw, cal, std

    def predict_pair(
        self,
        sequence: str,
        target_id: str,
        net_charge: float | None = None,
    ) -> dict[str, Any]:
        """Prediz alta atividade (MIC ≤ 3,4 µM) para um par peptídeo × alvo."""
        x, _, feats = self._feature_vector(sequence, target_id, net_charge=net_charge)
        raw, cal, std = self._raw_and_calibrated_prob(x)
        lo = max(0.0, cal - std)
        hi = min(1.0, cal + std)
        out = {
            **feats,
            "pred_high_activity_prob_raw": raw,
            "pred_high_activity_prob": cal,
            "pred_uncertainty_std": std,
            "pred_interval_low": lo,
            "pred_interval_high": hi,
            "calibrated": self._calibrator is not None,
        }
        return {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in out.items()
        }

    def find_neighbors(
        self,
        sequence: str,
        k: int = 5,
        target_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vizinhos do treino por identidade de sequência (+ cosine ESM se disponível).

        Score: ``0.6 * identidade + 0.4 * cosine`` (cai para identidade se não houver emb).
        """
        if self._mic_index.empty:
            return []
        seq = "".join(c for c in sequence.upper() if c.isalpha())
        # Uma linha por peptídeo com estatísticas agregadas dos MICs
        uniq = (
            self._mic_index.groupby("peptide_id", as_index=False)
            .agg(
                sequence=("sequence", "first"),
                name=("name", "first"),
                n_mic=("mic_value", "count"),
                mic_median=("mic_value", "median"),
                mic_min=("mic_value", "min"),
                frac_high=("mic_value", lambda s: float((s <= 3.4).mean())),
            )
        )
        q_emb = None
        try:
            q_emb = self.embed_sequence(seq)
            q_norm = np.linalg.norm(q_emb) + 1e-9
        except Exception:
            q_emb = None

        rows = []
        for _, r in uniq.iterrows():
            other = str(r["sequence"]).upper()
            if other == seq:
                continue
            ident = _seq_identity(seq, other)
            cosine = None
            if q_emb is not None and other in self._embeddings_cache:
                o = self._embeddings_cache[other]
                cosine = float(np.dot(q_emb, o) / (q_norm * (np.linalg.norm(o) + 1e-9)))
            score = 0.6 * ident + 0.4 * (cosine if cosine is not None else ident)
            row = {
                "peptide_id": r["peptide_id"],
                "name": r["name"],
                "sequence": other,
                "identity": round(ident, 3),
                "embedding_cosine": None if cosine is None else round(cosine, 3),
                "neighbor_score": round(score, 3),
                "n_mic": int(r["n_mic"]),
                "mic_median_uM": round(float(r["mic_median"]), 3),
                "mic_min_uM": round(float(r["mic_min"]), 3),
                "frac_high_activity": round(float(r["frac_high"]), 3),
            }
            if target_id:
                sub = self._mic_index[
                    (self._mic_index["peptide_id"] == r["peptide_id"])
                    & (self._mic_index["target_id"] == target_id)
                ]
                if not sub.empty:
                    row["mic_on_target_uM"] = round(float(sub["mic_value"].iloc[0]), 3)
            rows.append(row)

        rows.sort(key=lambda d: d["neighbor_score"], reverse=True)
        return rows[:k]

    def _feature_vector(
        self,
        sequence: str,
        target_id: str,
        net_charge: float | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Monta (x, nomes, dict de features) para o par informado."""
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
        """Predição + contribuições SHAP locais (fundo = matriz MIC de treino)."""
        from pepmem.shap_explain import explain_instance, load_training_matrix

        x, names, feats = self._feature_vector(sequence, target_id, net_charge=net_charge)
        try:
            X_bg, _, _ = load_training_matrix(self.use_embeddings)
            bg = self._model.named_steps["scaler"].transform(X_bg)
        except ValueError:
            bg = None

        explanation = explain_instance(self._model, x, names, background=bg)
        raw, cal, std = self._raw_and_calibrated_prob(x)
        return {
            **feats,
            "pred_high_activity_prob_raw": raw,
            "pred_high_activity_prob": cal,
            "pred_uncertainty_std": std,
            "expected_value_logit": explanation["expected_value_logit"],
            "shap_contributions": explanation["contributions"],
            "calibrated": self._calibrator is not None,
        }

    def global_shap_report(self) -> dict[str, Any] | None:
        """Lê o relatório global SHAP persistido em ``data/processed/models/``."""
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
        """Rankeia alvos: prob. calibrada − λ · toxicidade proxy (célula normal).

        ``pmi_sel`` (PMI vs membrana normal) entra como bônus leve.
        """
        ids = target_ids or self.targets["target_id"].tolist()
        rows = [self.predict_pair(sequence, tid, net_charge=net_charge) for tid in ids]
        df = pd.DataFrame(rows)

        # --- seletividade PMI e score final ---
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
        """Lista alvos disponíveis para o seletor da UI."""
        return self.targets[
            ["target_id", "target", "target_type", "surface_charge", "anionic_fraction"]
        ].to_dict(orient="records")

    @property
    def model_info(self) -> dict[str, Any]:
        """Metadados LOO/LOPO e calibração do modelo ativo (JSON de treino)."""
        meta_path = ROOT / "data" / "processed" / "models" / "multimodal_mic_loo.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        meta_path = ROOT / "data" / "processed" / "models" / "baseline_mic_loo.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}
