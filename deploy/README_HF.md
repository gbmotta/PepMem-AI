---
title: PepMem-AI
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.41.0"
app_file: dashboard/app.py
pinned: false
license: mit
---

# PepMem-AI

PoC de **predição peptídeo–membrana** (InovAI Lab / UFRN).

- **Predição** — PMI + probabilidade de alta atividade (MIC)
- **Ranking** — compara um peptídeo em várias membranas-alvo
- **XAI (SHAP)** — beeswarm + explicações locais

**Exemplo:** StigA6 `FFSLIPKLVKGLISAFK` · Stigmurin `FFSLIPSLVGGLISAFK`

> Modelo treinado com dados preliminares (12 MICs). Use PMI_sel como critério complementar.
