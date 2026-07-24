---
title: PepMem-AI
emoji: 🧬
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# PepMem-AI

Predição **peptídeo–membrana** (InovAI Lab / UFRN).

- **Predição** — PMI + probabilidade de alta atividade (MIC)
- **Ranking** — multi-alvo com score de priorização
- **XAI (SHAP)** — beeswarm + explicações locais
- **Narrativa** — “Explicar em português” (template; Qwen GGUF opcional, sem API)
- Atalhos: peptídeos **no banco** e **fora do treino**

**Exemplos:** StigA6 `FFSLIPKLVKGLISAFK` · mutante novo `FFSLIPKLVAGLISAFK`

> Treino atual: ~90 MICs (literatura + bancada). Use PMI_sel junto com a probabilidade do RF.

### Qwen GGUF (opcional, local)

A narrativa **não altera** PMI/probabilidade. Sem modelo, usa template.

1. Instale `llama-cpp-python` (opcional: `pip install llama-cpp-python==0.3.4`).
2. Coloque um GGUF em `models/llm/` (ex.: `qwen2.5-0.5b-instruct-q4_k_m.gguf`) **ou**
3. Defina `PEPMEM_GGUF_PATH` / `PEPMEM_LLM_AUTO_DOWNLOAD=1` nos Secrets do Space.
