# Modelos locais (opcional)

Coloque GGUF de narrativa em `models/llm/` (pasta ignorada pelo git), por exemplo:

- `qwen2.5-0.5b-instruct-q4_k_m.gguf`  
  (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`)

Ou defina `PEPMEM_GGUF_PATH`. Para baixar no primeiro uso no Space: `PEPMEM_LLM_AUTO_DOWNLOAD=1`.

Sem GGUF / sem `llama-cpp-python`, o botão **Explicar** usa só o **template** — as predições do RF não mudam.
