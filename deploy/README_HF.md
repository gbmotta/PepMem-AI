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

Predição **peptídeo–membrana** (InovAI Lab / UFRN · *Tityus stigmurus*).

| Aba | Função |
|-----|--------|
| Predição | PMI + prob. calibrada · lote FASTA · relatório MD/DOCX/PDF |
| Ranking | Multi-alvo + explicação + relatório |
| XAI (SHAP) | Global, beeswarm, local · texto sobre SHAP/ESM-2 |
| Datasets | Peptídeos do projeto |

**Exemplos:** StigA6 `FFSLIPKLVKGLISAFK` · mutante `FFSLIPKLVAGLISAFK`

> Treino ~90 MICs (literatura + bancada). Cloud = baseline; Space = multimodal (ESM-2).  
> Código e docs: [github.com/gbmotta/PepMem-AI](https://github.com/gbmotta/PepMem-AI) · guia: `docs/DEPLOY.md`
