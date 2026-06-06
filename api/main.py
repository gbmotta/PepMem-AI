"""PepMem-AI REST API (PoC)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pepmem.predictor import PepMemPredictor

_predictor: PepMemPredictor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _predictor
    _predictor = PepMemPredictor(use_embeddings=True)
    yield
    _predictor = None


app = FastAPI(
    title="PepMem-AI API",
    description="PoC: predição de interação peptídeo–membrana",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=200, examples=["FFSLIPKLVKGLISAFK"])
    target_id: str = Field(..., examples=["S_aureus_ATCC29213"])
    net_charge: Optional[float] = None


class RankRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=200)
    target_ids: Optional[list[str]] = None
    net_charge: Optional[float] = None
    lambda_tox: float = 0.5


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/targets")
def list_targets():
    assert _predictor
    return {"targets": _predictor.list_targets()}


@app.get("/model/info")
def model_info():
    assert _predictor
    return _predictor.model_info


@app.post("/predict")
def predict(body: PredictRequest):
    assert _predictor
    try:
        result = _predictor.predict_pair(body.sequence, body.target_id, net_charge=body.net_charge)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/explain")
def explain(body: PredictRequest):
    assert _predictor
    try:
        result = _predictor.explain_pair(body.sequence, body.target_id, net_charge=body.net_charge)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/explain/global")
def explain_global():
    assert _predictor
    report = _predictor.global_shap_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Relatório SHAP global ausente. Execute scripts/compute_shap.py.",
        )
    return report


@app.post("/rank")
def rank(body: RankRequest):
    assert _predictor
    try:
        df = _predictor.rank_peptide(
            body.sequence,
            target_ids=body.target_ids,
            net_charge=body.net_charge,
            lambda_tox=body.lambda_tox,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": df.to_dict(orient="records")}
