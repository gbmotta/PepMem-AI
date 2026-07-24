# Modelos locais (opcional)

## Qwen GGUF (narrativa)

A explicação em português do dashboard usa **template** por padrão. Para Qwen local (sem API):

1. `pip install llama-cpp-python==0.3.4`
2. Coloque um GGUF em `models/llm/` (ex.: `qwen2.5-0.5b-instruct-q4_k_m.gguf` de `Qwen/Qwen2.5-0.5B-Instruct-GGUF`)
   **ou** defina `PEPMEM_GGUF_PATH` / `PEPMEM_LLM_AUTO_DOWNLOAD=1`

A narrativa **não altera** PMI nem probabilidade do RF.

## Embeddings ESM-2

Gerados por `scripts/generate_embeddings.py` → `data/processed/embeddings/esm2_all.npz`  
Modelo: `facebook/esm2_t6_8M_UR50D` (ver `docs/TREINO.md` e a aba XAI do dashboard).
