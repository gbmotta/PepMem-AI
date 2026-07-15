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
- Atalhos: peptídeos **no banco** e **fora do treino**

**Exemplos:** StigA6 `FFSLIPKLVKGLISAFK` · mutante novo `FFSLIPKLVAGLISAFK`

> Treino atual: ~90 MICs (literatura + bancada). Use PMI_sel junto com a probabilidade do RF.
