#!/usr/bin/env python3
"""Generate ESM-2 embeddings for PepMem peptides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


def load_peptides(scope: str) -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")
    if scope == "project":
        return base[base["dataset"] == "PepMem-Base-Project"].copy()
    return base.dropna(subset=["sequence"]).copy()


def embed_batch(model, tokenizer, sequences: list[str], device: torch.device) -> np.ndarray:
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Média por sequência (ignora padding via atenção ao mask)
    mask = inputs["attention_mask"].unsqueeze(-1)
    hidden = outputs.last_hidden_state
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["project", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    out_dir = ROOT / "data" / "processed" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    peptides = load_peptides(args.scope)
    sequences = peptides["sequence"].tolist()
    ids = peptides["peptide_id"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {args.model} | device: {device} | peptídeos: {len(sequences)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    all_emb: list[np.ndarray] = []
    for i in tqdm(range(0, len(sequences), args.batch_size), desc="ESM-2"):
        batch = sequences[i : i + args.batch_size]
        all_emb.append(embed_batch(model, tokenizer, batch, device))

    matrix = np.vstack(all_emb)
    tag = args.scope
    np.savez_compressed(out_dir / f"esm2_{tag}.npz", peptide_ids=np.array(ids), embeddings=matrix)

    meta = {
        "model": args.model,
        "scope": args.scope,
        "n_peptides": len(ids),
        "embedding_dim": matrix.shape[1],
    }
    (out_dir / f"esm2_{tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Índice parquet (sem vetor completo — vetor fica no npz)
    index_df = peptides[["peptide_id", "sequence", "dataset"]].copy()
    index_df["embedding_file"] = f"esm2_{tag}.npz"
    index_df.to_parquet(out_dir / f"esm2_{tag}_index.parquet", index=False)
    print(f"Embeddings salvos: {out_dir / f'esm2_{tag}.npz'} ({matrix.shape})")


if __name__ == "__main__":
    main()
