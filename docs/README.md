# Documentação PepMem-AI

Índice dos materiais do projeto (código: `pepmem/`, `scripts/`, `api/`, `dashboard/`, `data/`).

| Pasta / arquivo | Conteúdo |
|-----------------|----------|
| [DEPLOY.md](DEPLOY.md) | Publicação (HF Spaces, Streamlit Cloud, Render) |
| [INTERPRETACAO_RESULTADOS.md](INTERPRETACAO_RESULTADOS.md) | Como ler PMI, probabilidades, ranking e SHAP |
| [TREINO.md](TREINO.md) | Dados, LOPO, calibração e artefatos |
| [exports/](exports/) | Mesmos guias em **PDF** e **DOCX** (`python scripts/export_docs.py`) |
| [peptideos/](peptideos/) | Documento consolidado (físico-química + MICs) |
| [pipeline/](pipeline/) | Slides Beamer + **banner A0** de congresso (modelagem/dados/resultados) |
| [proposta/](proposta/) | Proposta CNPq (PDF) |
| [referencias/](referencias/) | Artigos e teses de apoio |

## Dependências

| Arquivo | Uso |
|---------|-----|
| `requirements.txt` | Streamlit Cloud / Render (leve, **sem** PyTorch) |
| `requirements-space.txt` | HF Spaces / multimodal (`-r requirements.txt` + torch CPU) |
| `requirements-dev.txt` | Local completo (space + FastAPI) |

## Deploy HF — `deploy/README_HF.md`

**Não é guia redundante:** o script `scripts/deploy_hf_space.py` copia esse arquivo como `README.md` do Space. Ele precisa do **frontmatter YAML** (título, emoji, `sdk: docker`). O conteúdo longo de deploy fica em [`DEPLOY.md`](DEPLOY.md); GGUF opcional em [`../models/README.md`](../models/README.md).
