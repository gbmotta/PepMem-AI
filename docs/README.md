# Documentação PepMem-AI

Índice dos materiais do projeto (código: `pepmem/`, `scripts/`, `api/`, `dashboard/`, `data/`).

| Pasta / arquivo | Conteúdo |
|-----------------|----------|
| [DEPLOY.md](DEPLOY.md) | Guia de publicação (HF Spaces, Streamlit Cloud, Render) |
| [INTERPRETACAO_RESULTADOS.md](INTERPRETACAO_RESULTADOS.md) | Como ler PMI, probabilidades, intervalo, ranking e SHAP |
| [TREINO.md](TREINO.md) | Como os modelos foram treinados (dados, LOPO, calibração) |
| [peptideos/](peptideos/) | Documento consolidado (físico-química + MICs + mapa de estudos) |
| [pipeline/](pipeline/) | Entregáveis LaTeX / PDF do pipeline InovAI |
| [proposta/](proposta/) | Proposta CNPq (PDF) |
| [referencias/](referencias/) | Artigos e teses de apoio (ex.: Parente 2022) |

## Dependências

| Arquivo | Uso |
|---------|-----|
| `requirements.txt` | Streamlit Cloud / Render (leve, **sem** PyTorch) |
| `requirements-space.txt` | HF Spaces / multimodal (`-r requirements.txt` + torch CPU) |
| `requirements-dev.txt` | Local completo (space + FastAPI) |

Qwen GGUF opcional: `pip install llama-cpp-python==0.3.4` (ver `deploy/README_HF.md`).

Scripts de staging Hugging Face: `deploy/README_HF.md` (usado por `scripts/deploy_hf_space.py`).
