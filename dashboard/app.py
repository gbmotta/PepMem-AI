"""Dashboard Streamlit do PepMem-AI — layout estilo Power BI.

Relatório analítico: barra de relatório · painel de filtros · páginas (abas)
· faixa de KPIs · tiles de visuais. Paleta peçonha / peptídeo / membrana.

Execução local:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud adiciona só ``dashboard/`` ao path — a raiz do repo precisa entrar antes
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pepmem.predictor import PepMemPredictor, torch_available
from pepmem.shap_explain import plot_beeswarm, plot_contributions, plot_global_importance
from pepmem.paths import project_root

# Sempre ao lado de app.py no Cloud (evita ImportError de pepmem.narrative)
from narrative_lib import (  # type: ignore  # noqa: E402
    llm_status,
    narrate_batch,
    narrate_ranking,
    narrate_shap_overview,
    narrate_single,
)
from report_export import (  # type: ignore  # noqa: E402
    build_batch_report,
    build_ranking_report,
    build_shap_overview_report,
    build_single_report,
    export_report_bundle,
)

# Garante ROOT consistente com o restante do pacote
ROOT = project_root()

# --- paleta (carapaça · veneno · verde · roxo) ---
PM_VENOM = "#d4a017"
PM_GREEN = "#2f7d4a"
PM_PURPLE = "#5b3d8a"
PM_MEMBRANE = "#1e5c5a"
PM_CARAPACE = "#1c1410"

ASSETS = Path(__file__).resolve().parent / "assets"
THEME_CSS = ASSETS / "theme_pepmem.css"

st.set_page_config(
    page_title="PepMem-AI · Relatório",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Injeta CSS Power BI + paleta peçonha."""
    css = THEME_CSS.read_text(encoding="utf-8") if THEME_CSS.exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def report_bar(subtitulo: str, n_mics: int, lope_auc: float | None) -> None:
    """Barra superior estilo relatório Power BI."""
    auc_txt = f"{lope_auc:.3f}" if lope_auc is not None else "—"
    st.markdown(
        f"""
        <div class="pbi-bar">
          <div>
            <div class="brand">InovAI Lab · UFRN · Tityus stigmurus</div>
            <div class="title">PepMem-AI — Interação peptídeo–membrana</div>
          </div>
          <div class="meta">
            {html.escape(subtitulo)}<br/>
            Treino <strong>{n_mics} MICs</strong>
            · Leave-peptide AUC <strong>{auc_txt}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict], cols: int | None = None) -> None:
    """Faixa de cards KPI (tile Power BI)."""
    n = cols or len(items)
    cells = []
    for i, it in enumerate(items):
        cls = "pbi-kpi"
        tone = it.get("tone")
        if tone == "warn":
            cls += " warn"
        elif tone == "ok":
            cls += " ok"
        elif tone == "membrane":
            cls += " membrane"
        elif tone == "purple":
            cls += " purple"
        hint = (
            f'<div class="hint">{html.escape(it.get("hint", ""))}</div>'
            if it.get("hint")
            else ""
        )
        cells.append(
            f'<div class="{cls}" style="animation-delay:{0.04 * i}s">'
            f'<div class="label">{html.escape(it["label"])}</div>'
            f'<div class="value">{html.escape(str(it["value"]))}</div>'
            f"{hint}</div>"
        )
    st.markdown(
        f'<div class="pbi-kpi-row" style="grid-template-columns:repeat({n},minmax(0,1fr))">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def tile_title(title: str, subtitle: str = "") -> None:
    """Título de visual / tile."""
    sub = f'<div class="pbi-tile-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="pbi-tile-title">{html.escape(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def filter_label(text: str) -> None:
    st.markdown(f'<div class="pbi-filter-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def info_box(title: str, body_html: str) -> None:
    """Caixa curta de orientação (como funciona / interpretação)."""
    st.markdown(
        f'<div class="pm-hint-box"><strong>{html.escape(title)}</strong><br/>{body_html}</div>',
        unsafe_allow_html=True,
    )


# --- presets ---
# --- exemplos opcionais (um só lugar no formulário) ---
PRESETS = [
    ("— digitar / colar / FASTA —", "", None),
    ("StigA6 (conhecido do projeto)", "FFSLIPKLVKGLISAFK", 3.0),
    ("Stigmurin (conhecido do projeto)", "FFSLIPSLVGGLISAFK", 3.0),
    ("StigA16 (conhecido do projeto)", "FFKLIPKLVKGLISAFK", 4.0),
    ("Mutante S→A (sequência nova)", "FFSLIPKLVAGLISAFK", 3.0),
]


@st.cache_resource
def get_predictor() -> PepMemPredictor:
    """Singleton do predictor. No Cloud (sem torch) usa só o baseline."""
    return PepMemPredictor(use_embeddings=torch_available())


@st.cache_data
def load_project_peptides() -> pd.DataFrame:
    """Catálogo PepMem-Base-Project (IDs PXX e sequências de referência)."""
    path = ROOT / "data" / "processed" / "pepmem_base_project.parquet"
    if not path.exists():
        # fallback legado
        csv_path = ROOT / "data" / "processed" / "pepmem_base_project.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Gerando beeswarm SHAP...")
def cached_beeswarm(use_embeddings: bool, n_mic: int, _layout_version: int = 7) -> bytes:
    """PNG do beeswarm: prefere artefato offline; regenera se necessário."""
    kind = "multimodal" if use_embeddings else "baseline"
    static = ROOT / "data" / "processed" / "models" / f"shap_beeswarm_{kind}.png"
    if static.exists():
        return static.read_bytes()

    import io

    import joblib
    import matplotlib.pyplot as plt

    fname = "multimodal_mic_rf.joblib" if use_embeddings else "baseline_mic_rf.joblib"
    path = ROOT / "data" / "processed" / "models" / fname
    if not path.exists():
        raise FileNotFoundError(f"Modelo ausente: {fname}")
    pipe = joblib.load(path)
    fig = plot_beeswarm(pipe, use_embeddings, title=f"Beeswarm SHAP — {kind} ({n_mic} MICs)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="Calculando SHAP...")
def cached_explain(
    sequence: str,
    target_id: str,
    net_charge: float | None,
    _shap_fix: int = 3,
) -> dict:
    """SHAP local cacheado por (sequência, alvo, carga)."""
    return get_predictor().explain_pair(sequence, target_id, net_charge=net_charge)


inject_theme()

# --- estado compartilhado ---
predictor = get_predictor()
targets = predictor.targets
target_options = targets.set_index("target_id")["target"].to_dict()
project_df = load_project_peptides()
seq_to_project = (
    {str(r["sequence"]).upper(): r for _, r in project_df.dropna(subset=["sequence"]).iterrows()}
    if not project_df.empty
    else {}
)

info = predictor.model_info or {}
n_train = int(info.get("n_samples") or 90)
loo_auc = info.get("loo_auc")
lope = info.get("leave_one_peptide_auc")
HAS_TORCH = torch_available()


def format_target_label(target_id: str) -> str:
    """Rótulo curto do alvo para selectbox / títulos."""
    name = target_options.get(target_id, target_id)
    return name if len(name) <= 40 else f"{name[:37]}…"


# Rótulos legíveis para colunas de tabelas (variável → texto comum)
COLUMN_LABELS: dict[str, str] = {
    "peptide_id": "ID do peptídeo",
    "name": "Nome",
    "sequence": "Sequência",
    "header": "Cabeçalho",
    "identity": "Identidade",
    "neighbor_score": "Score do vizinho",
    "mic_median_uM": "MIC mediana (µM)",
    "frac_high_activity": "Fração alta atividade",
    "mic_alvo": "MIC no alvo (µM)",
    "mic_on_target_uM": "MIC no alvo (µM)",
    "net_charge": "Carga líquida",
    "source": "Origem",
    "origem": "Origem",
    "nota": "Nota",
    "aa": "Aminoácidos",
    "length": "Comprimento",
    "pmi": "PMI",
    "pmi_sel": "PMI seletivo",
    "q_peptide": "Carga do peptídeo",
    "prob_calibrada": "Prob. calibrada",
    "pred_high_activity_prob": "Prob. alta atividade",
    "faixa": "Faixa",
    "no_banco": "No banco",
    "erro": "Erro",
    "target_id": "Alvo",
    "target_type": "Tipo de alvo",
    "final_score": "Score final",
    "label": "Descritor",
    "shap_value": "Valor SHAP",
    "group": "Grupo",
}


def truncate_text(value: object, max_len: int = 40) -> object:
    """Trunca strings longas em tabelas densas."""
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def humanize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas técnicas para rótulos legíveis no dashboard."""
    rename = {c: COLUMN_LABELS[c] for c in df.columns if c in COLUMN_LABELS}
    return df.rename(columns=rename)


def _prob_as_float(value: object) -> float | None:
    """Converte célula de probabilidade (float ou '70.0%') para [0, 1]."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def show_table(
    df: pd.DataFrame,
    max_text_len: int = 40,
    *,
    highlight_col: str | None = None,
    highlight_gte: float = 0.70,
) -> None:
    """Tabela truncada com cabeçalhos legíveis; opcionalmente destaca linhas ≥ limiar."""
    raw = df.copy()
    view = humanize_columns(raw)
    for col in view.columns:
        if view[col].dtype == object:
            view[col] = view[col].map(lambda x: truncate_text(x, max_text_len))

    tech_col = None
    if highlight_col and highlight_col in raw.columns:
        tech_col = highlight_col
    elif highlight_col:
        inv = {v: k for k, v in COLUMN_LABELS.items()}
        cand = inv.get(highlight_col)
        if cand and cand in raw.columns:
            tech_col = cand

    if tech_col is not None:
        probs = raw[tech_col].map(_prob_as_float)

        def _style_row(row: pd.Series) -> list[str]:
            prob = probs.loc[row.name] if row.name in probs.index else None
            if prob is not None and prob >= highlight_gte:
                return ["background-color: rgba(47, 125, 74, 0.16)"] * len(row)
            return [""] * len(row)

        st.dataframe(view.style.apply(_style_row, axis=1), use_container_width=True, hide_index=True)
    else:
        st.dataframe(view, use_container_width=True, hide_index=True)


def show_png(data) -> None:
    """Exibe PNG: Streamlit 1.37 usa use_column_width; versões novas usam use_container_width."""
    try:
        st.image(data, use_column_width=True)
    except TypeError:
        st.image(data, use_container_width=True)


def activity_band(prob: float) -> tuple[str, str, str]:
    """Faixa de interpretação da prob. calibrada → (classe CSS, título, texto)."""
    if prob >= 0.70:
        return (
            "high",
            "Alta confiança de atividade",
            "O modelo associa este par a MIC tipicamente ≤ 3,4 µM. Priorize para ensaio, mas confirme in vitro.",
        )
    if prob >= 0.40:
        return (
            "mid",
            "Candidato intermediário",
            "Nem claramente ativo nem inativo. Use PMI/PMI_sel e compare com análogos do projeto no ranking.",
        )
    return (
        "low",
        "Baixa probabilidade de atividade",
        "Padrão mais próximo de nativos fracos no treino. Pode ainda ter PMI útil; valide com cuidado.",
    )


def lookup_sequence(seq: str) -> dict | None:
    """Retorna registro do projeto se a sequência já estiver no banco de treino."""
    key = "".join(c for c in seq.upper() if c.isalpha())
    return seq_to_project.get(key)


def render_result_banner(prob: float, in_db: dict | None) -> None:
    """Resultado herói: probabilidade grande + faixa + badge banco."""
    css, title, msg = activity_band(prob)
    if in_db is not None:
        badge = (
            f'<span class="pm-badge in">No banco · '
            f'{html.escape(str(in_db.get("peptide_id")))} · '
            f'{html.escape(str(in_db.get("name")))}</span>'
        )
    else:
        badge = '<span class="pm-badge out">Fora do treino · predição generalizada</span>'
    st.markdown(
        f'<div class="pm-hero {css}">'
        f'<div class="pm-hero-prob">{prob:.0%}</div>'
        f'<div class="pm-hero-body">{badge}'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(msg)}</p></div></div>",
        unsafe_allow_html=True,
    )


def empty_cta(title: str, body: str) -> None:
    """Estado vazio com CTA claro (o que fazer agora)."""
    st.markdown(
        f'<div class="pm-empty"><strong>{html.escape(title)}</strong>'
        f"<p>{html.escape(body)}</p></div>",
        unsafe_allow_html=True,
    )


def _report_fingerprint(report) -> str:
    """Hash estável do conteúdo (ignora generated_at) para cache de export."""
    parts: list[str] = [str(report.title), str(report.subtitle)]
    for sec in report.sections:
        parts.append(str(sec.title))
        parts.extend(str(x) for x in (sec.paragraphs or []))
        parts.extend(str(x) for x in (sec.bullets or []))
        if sec.table_headers:
            parts.extend(str(x) for x in sec.table_headers)
        if sec.table_rows:
            for row in sec.table_rows:
                parts.extend(str(c) for c in row)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def render_shap_block(expl: dict, target_label: str) -> None:
    """Métricas + gráfico de contribuições SHAP locais."""
    kpi_row(
        [
            {
                "label": "Prob. alta atividade",
                "value": f"{expl['pred_high_activity_prob']:.1%}",
                "tone": "ok" if expl["pred_high_activity_prob"] >= 0.7 else None,
                "hint": "calibrada (isotonic LOPO)",
            }
        ],
        cols=1,
    )
    fig = plot_contributions(
        expl["shap_contributions"],
        title=f"SHAP — {target_label}",
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    # Sem expander aqui: este bloco também roda dentro de outro expander (Cloud proíbe aninhamento)
    st.caption("Tabela de contribuições")
    df = pd.DataFrame(expl["shap_contributions"])[["label", "shap_value", "group"]]
    df["shap_value"] = df["shap_value"].map(lambda x: f"{x:+.4f}")
    show_table(df, max_text_len=36)
    st.caption("Valores positivos → favorecem alta atividade · negativos → desfavorecem")


def render_narrative_box(text: str, source: str) -> None:
    """Mostra texto de explicação (não altera números da predição)."""
    st.markdown(
        f'<div class="pm-hint-box"><strong>Explicação</strong><br/>'
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_report_downloads(report, key_prefix: str, filename_stem: str) -> None:
    """Um seletor de formato + download; bundle cacheado por fingerprint do conteúdo."""
    fp = _report_fingerprint(report)
    key_slot = f"_bundle_key_{key_prefix}"
    cache_slot = f"_bundle_{key_prefix}"
    if st.session_state.get(key_slot) != fp:
        try:
            st.session_state[cache_slot] = export_report_bundle(report)
            st.session_state[key_slot] = fp
        except Exception as e:
            st.warning(f"Não foi possível gerar os arquivos do relatório: {e}")
            return
    bundle = st.session_state[cache_slot]

    fmt_labels = {
        "PDF (.pdf)": ("pdf", "pdf", "application/pdf"),
        "Word (.docx)": (
            "docx",
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        "Markdown (.md)": ("md", "md", "text/markdown"),
    }
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        fmt = st.selectbox(
            "Formato do relatório",
            options=list(fmt_labels.keys()),
            key=f"{key_prefix}_fmt",
        )
    bundle_key, ext, mime = fmt_labels[fmt]
    with c2:
        st.write("")  # alinha verticalmente com o selectbox
        st.download_button(
            f"Baixar relatório ({ext.upper()})",
            data=bundle[bundle_key],
            file_name=f"{filename_stem}.{ext}",
            mime=mime,
            use_container_width=True,
            key=f"{key_prefix}_dl",
        )


def narrative_engine_caption() -> None:
    """Status curto do motor de narrativa (opcional)."""
    st_stat = llm_status()
    if st_stat["gguf_ready"] and st_stat["llama_cpp_installed"]:
        st.caption("Narrativa com Qwen GGUF local disponível.")
    elif st_stat["llama_cpp_installed"]:
        st.caption(
            "Para Qwen local, coloque um `.gguf` em `models/llm/` "
            "ou defina `PEPMEM_GGUF_PATH`."
        )


def apply_preset(seq: str, charge: float) -> None:
    """Callback on_click — define sessão antes dos widgets no próximo rerun."""
    st.session_state["seq_main"] = (seq or "").upper()
    st.session_state["charge_main"] = float(charge)
    st.session_state["use_charge_main"] = True


def apply_fasta_sequence(seq: str, header: str = "") -> None:
    """Callback: preenche a sequência a partir de um registro FASTA."""
    st.session_state["seq_main"] = (seq or "").upper()
    if header:
        st.session_state["fasta_header"] = header


def force_uppercase(key: str) -> None:
    """Callback on_change: força maiúsculas no campo de sequência."""
    val = st.session_state.get(key)
    if isinstance(val, str) and val != val.upper():
        st.session_state[key] = val.upper()


def force_fasta_seq_uppercase(key: str = "fasta_paste") -> None:
    """Maiúsculas só nas linhas de sequência (headers `>` preservados)."""
    val = st.session_state.get(key)
    if not isinstance(val, str):
        return
    lines = []
    for line in val.splitlines():
        if line.startswith(">"):
            lines.append(line)
        else:
            lines.append(line.upper())
    new = "\n".join(lines)
    if val.endswith("\n"):
        new += "\n"
    if new != val:
        st.session_state[key] = new


def clear_session_results() -> None:
    """Callback: limpa predições / narrativas / caches de relatório da sessão."""
    for k in (
        "last_single",
        "last_single_narrative",
        "last_batch",
        "last_batch_narrative",
        "last_shap_overview_narrative",
        "last_xai_local",
        "last_xai_narrative",
        "last_rank",
        "last_rank_narrative",
    ):
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if str(k).startswith("_bundle_") or str(k).startswith("_bundle_key_"):
            st.session_state.pop(k, None)


def _decode_upload_bytes(raw: bytes) -> str:
    """Decodifica bytes de upload (UTF-8 com fallback latin-1)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def load_fasta_from_text(text: str, source: str = "colar") -> list[dict]:
    """Parseia texto FASTA / multi-FASTA → registros com sequência válida."""
    from pepmem.peptide_utils import parse_fasta_text

    records = parse_fasta_text(text or "")
    out: list[dict] = []
    for r in records:
        if not r.get("sequence"):
            continue
        out.append(
            {
                "header": r.get("header") or "",
                "sequence": r["sequence"],
                "source": source,
            }
        )
    return out


def load_fasta_records(uploaded) -> list[dict]:
    """Lê um upload Streamlit (.fasta/.fa/.txt) e devolve registros parseados."""
    name = getattr(uploaded, "name", None) or "upload"
    return load_fasta_from_text(_decode_upload_bytes(uploaded.getvalue()), source=name)


def merge_fasta_uploads(uploads) -> list[dict]:
    """Junta um ou vários arquivos FASTA (cada um pode ser multi-FASTA)."""
    if uploads is None:
        return []
    files = uploads if isinstance(uploads, list) else [uploads]
    merged: list[dict] = []
    for f in files:
        merged.extend(load_fasta_records(f))
    return merged


# Defaults de sessão (antes dos widgets)
if "seq_main" not in st.session_state:
    st.session_state["seq_main"] = "FFSLIPKLVKGLISAFK"
if "charge_main" not in st.session_state:
    st.session_state["charge_main"] = 3.0
if "use_charge_main" not in st.session_state:
    st.session_state["use_charge_main"] = True

if "charge_batch" not in st.session_state:
    st.session_state["charge_batch"] = float(st.session_state.get("charge_main", 3.0))
if "use_charge_batch" not in st.session_state:
    st.session_state["use_charge_batch"] = False

# --- sidebar = painel útil ---
with st.sidebar:
    st.markdown("### PepMem-AI")
    st.caption("InovAI Lab · UFRN · *Tityus stigmurus*")

    modo = "Multimodal (ESM-2)" if HAS_TORCH else "Baseline (Cloud)"
    st.markdown(f"**{modo}**")
    meta_bits = [f"{n_train} MICs"]
    if lope is not None:
        meta_bits.append(f"AUC LOPO {float(lope):.3f}")
    st.caption(" · ".join(meta_bits))
    st.caption("Alta atividade = MIC ≤ 3,4 µM")

    st.markdown("---")
    st.markdown("##### Como ler o resultado")
    st.markdown(
        "- **≥ 70%** → priorizar ensaio\n"
        "- **40–70%** → intermediário\n"
        "- **< 40%** → baixa chance\n"
        "- **PMI** alto → boa interação estimada\n"
        "- Use **intervalo** e **vizinhos** para calibrar confiança"
    )

    st.markdown("---")
    st.markdown("##### Atalho de exemplo")
    side_presets = [p for p in PRESETS if p[1]]
    side_labels = [p[0] for p in side_presets]
    side_pick = st.selectbox(
        "Carregar no formulário",
        options=side_labels,
        key="sidebar_preset",
        label_visibility="collapsed",
    )
    _, pseq, pch = next(p for p in side_presets if p[0] == side_pick)
    st.button(
        "Aplicar exemplo",
        use_container_width=True,
        key="sidebar_apply_preset",
        on_click=apply_preset,
        args=(pseq, float(pch)),
    )

    st.markdown("---")
    st.markdown("##### Onde ir")
    st.markdown(
        "- **Predição** — 1 peptídeo ou lote FASTA\n"
        "- **Ranking** — multi-alvo + relatório\n"
        "- **XAI** — SHAP / beeswarm\n"
        "- **Dados** — peptídeos do projeto · API local"
    )

    last = st.session_state.get("last_single")
    if last:
        st.markdown("---")
        st.markdown("##### Última predição")
        res = last["res"]
        prob = float(res["pred_high_activity_prob"])
        seq = last["sequence"]
        seq_show = seq if len(seq) <= 18 else f"{seq[:15]}…"
        st.caption(f"`{seq_show}`")
        st.caption(last.get("target_label", last.get("target_id", ""))[:48])
        st.metric("Prob. calibrada", f"{prob:.0%}", help="Da última predição nesta sessão")
        st.caption(f"PMI {float(res['pmi']):.3f}")
        st.button(
            "Limpar resultado",
            use_container_width=True,
            key="sidebar_clear",
            on_click=clear_session_results,
        )

    st.markdown("---")
    with st.expander("Dica de lote"):
        st.caption(
            "Em Predição → Lote: envie um multi-FASTA ou vários arquivos, "
            "escolha a membrana e clique em Predizer todas."
        )
    with st.expander("Retreino com MIC novo"):
        st.caption(
            "Edite `data/bench/mic_bench.csv` e rode "
            "`python scripts/import_bench_mic.py --retrain`."
        )

# --- barra de relatório + KPIs globais ---
mode_label = (
    "PoC priorização in vitro · peçonha escorpiônica"
    if HAS_TORCH
    else "Modo Cloud (baseline + PMI) · multimodal no Space HF"
)
report_bar(
    mode_label,
    n_train,
    float(lope) if lope is not None else None,
)

kpi_row(
    [
        {
            "label": "MICs no treino",
            "value": str(n_train),
            "hint": "literatura + bancada",
            "tone": "membrane",
        },
        {
            "label": "Leave-peptide AUC",
            "value": f"{float(lope):.3f}" if lope is not None else "—",
            "hint": "validação principal",
            "tone": "ok" if lope is not None and float(lope) >= 0.8 else None,
        },
        {
            "label": "Peptídeos projeto",
            "value": str(len(project_df) if not project_df.empty else "—"),
            "hint": "Stigmurin / StigA / TsAP",
            "tone": "purple",
        },
        {
            "label": "Alvos membrana",
            "value": str(len(target_options)),
            "hint": "Gram+ · Gram− · fungo · célula",
            "tone": "membrane",
        },
    ],
    cols=4,
)

info_box(
    "Como funciona (em 3 passos)",
    "1) Você informa um peptídeo e uma membrana-alvo.<br/>"
    "2) O sistema calcula descritores (carga, hidrofobicidade, PMI) e estima a "
    "chance de <em>alta atividade</em> (MIC ≤ 3,4 µM).<br/>"
    "3) Use o resultado para <strong>priorizar</strong> ensaios in vitro — não substitui o laboratório.",
)

st.markdown('<div class="pm-nav-wrap">', unsafe_allow_html=True)
page = st.radio(
    "Seção",
    options=["Predição", "Ranking", "XAI", "Dados"],
    horizontal=True,
    key="main_nav",
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predição
# ---------------------------------------------------------------------------
if page == "Predição":
    info_box(
        "Como ler a predição",
        "<strong>≥ 70%</strong> → candidato forte · "
        "<strong>40–70%</strong> → intermediário · "
        "<strong>&lt; 40%</strong> → baixa chance.<br/>"
        "<strong>PMI</strong> alto sugere boa interação eletrostática/hidrofóbica. "
        "O <strong>intervalo</strong> mostra incerteza entre as árvores do modelo. "
        "Confirme sempre na bancada.",
    )

    sub_unica, sub_lote = st.tabs(["Única", "Lote"])

    preferred = "S_aureus_ATCC29213"
    keys = list(target_options.keys())
    idx_pref = keys.index(preferred) if preferred in keys else 0

    with sub_unica:
        with st.container(border=True):
            tile_title("Par peptídeo × membrana", "Uma sequência · uma membrana")
            st.caption(
                "Exemplos **conhecidos do projeto** já têm MIC no treino. "
                "Exemplos **novos** servem só para testar a predição."
            )

            preset_labels = [p[0] for p in PRESETS]
            chosen = st.selectbox(
                "Carregar exemplo (opcional)", options=preset_labels, key="preset_select"
            )
            if chosen != preset_labels[0]:
                _, pseq, pch = next(p for p in PRESETS if p[0] == chosen)
                st.button(
                    "Aplicar exemplo",
                    use_container_width=True,
                    key="pred_apply_preset",
                    on_click=apply_preset,
                    args=(pseq, float(pch)),
                )

            st.markdown(
                '<div class="pm-hint-box">'
                "<strong>Peptídeo novo com MIC da bancada?</strong> "
                "Edite <code>data/bench/mic_bench.csv</code> e rode "
                "<code>python scripts/import_bench_mic.py --retrain</code>."
                "</div>",
                unsafe_allow_html=True,
            )

            sequence = st.text_input(
                "Sequência (letra única)",
                key="seq_main",
                help="Cole a sequência de aminoácidos (minúsculas viram maiúsculas)",
                placeholder="Ex.: FFSLIPKLVAGLISAFK",
                on_change=force_uppercase,
                args=("seq_main",),
            )
            if st.session_state.get("fasta_header"):
                st.caption(f"FASTA: **{st.session_state['fasta_header']}**")
            hit = lookup_sequence(sequence or "")
            if hit is not None:
                st.caption(
                    f"Este peptídeo já está no projeto: **{hit.get('peptide_id')} · {hit.get('name')}**"
                )
            else:
                st.caption("Sequência nova para o modelo (predição generalizada).")

            c1, c2 = st.columns(2)
            with c1:
                use_charge = st.checkbox("Informar carga manualmente", key="use_charge_main")
            with c2:
                net_charge = st.number_input(
                    "Carga líquida",
                    step=1.0,
                    format="%.1f",
                    key="charge_main",
                    disabled=not use_charge,
                )
            charge = float(net_charge) if use_charge else None

            target_id = st.selectbox(
                "Membrana-alvo",
                options=keys,
                index=idx_pref,
                format_func=format_target_label,
                key="target_unica",
            )
            run_pred = st.button("Predizer", type="primary", use_container_width=True)

        if run_pred:
            with st.spinner("Descritores · PMI · modelo RF…"):
                try:
                    res = predictor.predict_pair(sequence, target_id, net_charge=charge)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

            neighbors = predictor.find_neighbors(sequence, k=5, target_id=target_id)
            shap_top = None
            try:
                expl = cached_explain(sequence, target_id, charge)
                shap_top = expl.get("shap_contributions")
            except Exception:
                expl = None

            lo = res.get("pred_interval_low")
            hi = res.get("pred_interval_high")
            interval = (
                f"{100 * float(lo):.0f}–{100 * float(hi):.0f}%"
                if lo is not None and hi is not None
                else "—"
            )
            st.session_state["last_single"] = {
                "res": res,
                "sequence": sequence,
                "target_id": target_id,
                "target_label": target_options[target_id],
                "charge": charge,
                "hit": hit,
                "neighbors": neighbors,
                "expl": expl,
                "interval": interval,
            }
            st.session_state.pop("last_single_narrative", None)

        if st.session_state.get("last_single"):
            snap = st.session_state["last_single"]
            res = snap["res"]
            sequence = snap["sequence"]
            target_id = snap["target_id"]
            hit = snap.get("hit")
            neighbors = snap.get("neighbors") or []
            expl = snap.get("expl")
            interval = snap.get("interval") or "—"
            prob = float(res["pred_high_activity_prob"])

            render_result_banner(prob, hit)
            kpi_row(
                [
                    {
                        "label": "PMI",
                        "value": f"{res['pmi']:.3f}",
                        "hint": "índice peptídeo–membrana",
                        "tone": "membrane",
                    },
                    {
                        "label": "Prob. calibrada",
                        "value": f"{prob:.1%}",
                        "hint": "isotonic LOPO",
                        "tone": "ok" if prob >= 0.7 else ("warn" if prob < 0.4 else None),
                    },
                    {
                        "label": "Intervalo (árvores)",
                        "value": interval,
                        "hint": f"σ = {float(res.get('pred_uncertainty_std') or 0):.3f}",
                    },
                    {
                        "label": "Carga (q)",
                        "value": f"{res['q_peptide']:.1f}",
                        "hint": "peptídeo catiônico",
                    },
                ],
                cols=4,
            )

            if res.get("pred_high_activity_prob_raw") is not None:
                st.caption(
                    f"Prob. bruta: {float(res['pred_high_activity_prob_raw']):.1%} · "
                    f"σ árvores: {float(res.get('pred_uncertainty_std') or 0):.3f}"
                )

            with st.container(border=True):
                tile_title("Explicação em português", "Priorização para bancada")
                narrative_engine_caption()
                if st.button("Explicar resultado", type="primary", key="btn_narrate_single"):
                    with st.spinner("Gerando explicação…"):
                        out = narrate_single(
                            sequence=sequence,
                            target_label=snap["target_label"],
                            prob=prob,
                            pmi=float(res["pmi"]),
                            interval=interval,
                            q_peptide=float(res["q_peptide"]),
                            neighbors=neighbors,
                            shap_top=expl.get("shap_contributions") if expl else None,
                            in_project=hit is not None,
                            prefer_llm=True,
                        )
                        st.session_state["last_single_narrative"] = out
                if st.session_state.get("last_single_narrative"):
                    out = st.session_state["last_single_narrative"]
                    render_narrative_box(out["text"], out["source"])

                narr = (st.session_state.get("last_single_narrative") or {}).get("text")
                shap_rows = expl.get("shap_contributions") if expl is not None else None
                report = build_single_report(
                    sequence=sequence,
                    target_label=snap["target_label"],
                    res=res,
                    narrative=narr,
                    neighbors=neighbors,
                    shap_top=shap_rows,
                    interval=interval,
                    in_project=hit is not None,
                )
                render_report_downloads(report, "single_report", "pepmem_relatorio_predicao")

            with st.container(border=True):
                tile_title("Vizinhos no treino", "Identidade + cosine ESM-2")
                st.caption(
                    "Peptídeos parecidos no treino ajudam a contextualizar o resultado."
                )
                if neighbors:
                    ndf = pd.DataFrame(neighbors)[
                        [
                            "peptide_id",
                            "name",
                            "identity",
                            "neighbor_score",
                            "mic_median_uM",
                            "frac_high_activity",
                        ]
                    ]
                    if "mic_on_target_uM" in neighbors[0]:
                        ndf["mic_alvo"] = [n.get("mic_on_target_uM") for n in neighbors]
                    show_table(ndf, max_text_len=28)
                    top = neighbors[0]
                    st.caption(
                        f"Mais próximo: **{top['peptide_id']}** ({top.get('name')}) · "
                        f"identidade {100 * top['identity']:.0f}% · "
                        f"MIC mediana {top['mic_median_uM']} µM"
                    )
                else:
                    st.caption("Sem índice MIC para vizinhos.")

            with st.expander("Detalhes da resposta"):
                st.code(res["sequence"], language=None)
                st.json({k: v for k, v in res.items() if k != "sequence"})

            with st.expander("Explicação SHAP (contribuições locais)"):
                if expl is not None:
                    render_shap_block(expl, snap["target_label"])
                else:
                    try:
                        expl2 = cached_explain(sequence, target_id, snap.get("charge"))
                        render_shap_block(expl2, snap["target_label"])
                    except Exception as e:
                        st.warning(f"SHAP indisponível: {e}")
        else:
            empty_cta(
                "Ainda sem predição nesta sessão",
                "Cole uma sequência (ou aplique um exemplo), escolha a membrana e clique em Predizer.",
            )

    with sub_lote:
        with st.container(border=True):
            tile_title("Lote FASTA", "Vários peptídeos · mesma membrana")
            st.caption(
                "Envie **um multi-FASTA**, **vários arquivos** `.fasta`/`.fa`, ou **cole** "
                "vários registros (`>header` + sequência)."
            )
            fasta_files = st.file_uploader(
                "Arquivo(s) FASTA",
                type=["fasta", "fa", "faa", "fna", "txt"],
                accept_multiple_files=True,
                help="Um multi-FASTA ou vários arquivos; cada >header vira uma predição",
                key="fasta_upload",
            )
            fasta_paste = st.text_area(
                "Ou cole multi-FASTA aqui",
                height=110,
                placeholder=">pep1\nFFSLIPKLVKGLISAFK\n>pep2\nGILGKLWEGVKSIF\n…",
                key="fasta_paste",
                on_change=force_uppercase,
                args=("fasta_paste",),
            )

            fasta_recs: list[dict] = []
            try:
                fasta_recs.extend(merge_fasta_uploads(fasta_files))
                if (fasta_paste or "").strip():
                    fasta_recs.extend(load_fasta_from_text(fasta_paste, source="colar"))
            except Exception as e:
                st.error(f"FASTA inválido: {e}")
                fasta_recs = []

            seen_seq: set[str] = set()
            deduped: list[dict] = []
            for r in fasta_recs:
                seq = r["sequence"]
                if seq in seen_seq:
                    continue
                seen_seq.add(seq)
                deduped.append(r)
            fasta_recs = deduped

            use_charge_batch = st.checkbox(
                "Fixar carga para todo o lote",
                key="use_charge_batch",
            )
            net_charge_batch = st.number_input(
                "Carga líquida (lote)",
                step=1.0,
                format="%.1f",
                key="charge_batch",
                disabled=not use_charge_batch,
            )
            batch_charge = float(net_charge_batch) if use_charge_batch else None

            target_id_lote = st.selectbox(
                "Membrana-alvo (lote)",
                options=keys,
                index=idx_pref,
                format_func=format_target_label,
                key="target_lote",
            )

            run_fasta_batch = False
            if fasta_recs:
                n_files = len(fasta_files) if fasta_files else 0
                src_note = []
                if n_files:
                    src_note.append(f"{n_files} arquivo(s)")
                if (fasta_paste or "").strip():
                    src_note.append("texto colado")
                st.success(
                    f"**{len(fasta_recs)}** peptídeo(s) prontos"
                    + (f" · {' + '.join(src_note)}" if src_note else "")
                )
                preview = pd.DataFrame(
                    [
                        {
                            "header": (r.get("header") or f"seq_{i+1}")[:48],
                            "sequence": r["sequence"],
                            "aa": len(r["sequence"]),
                            "origem": r.get("source", ""),
                        }
                        for i, r in enumerate(fasta_recs)
                    ]
                )
                show_table(preview, max_text_len=42)

                labels = [
                    f"{(r.get('header') or f'seq_{i+1}')[:40]} · {r['sequence'][:14]}"
                    + ("…" if len(r["sequence"]) > 14 else "")
                    for i, r in enumerate(fasta_recs)
                ]
                pick = st.selectbox(
                    "Usar uma sequência na aba Única (opcional)",
                    options=list(range(len(fasta_recs))),
                    format_func=lambda i: labels[i],
                    key="fasta_pick",
                )
                c_fa1, c_fa2 = st.columns(2)
                with c_fa1:
                    st.button(
                        "Usar esta sequência",
                        use_container_width=True,
                        on_click=apply_fasta_sequence,
                        args=(
                            fasta_recs[pick]["sequence"],
                            fasta_recs[pick].get("header") or "",
                        ),
                    )
                with c_fa2:
                    run_fasta_batch = st.button(
                        f"Predizer todas ({len(fasta_recs)})",
                        type="primary",
                        use_container_width=True,
                        key="fasta_batch_btn",
                    )
                st.session_state["fasta_records"] = fasta_recs
            else:
                if fasta_files or (fasta_paste or "").strip():
                    st.warning(
                        "Nenhuma sequência válida encontrada. Use headers `>` e aminoácidos A–Y."
                    )
                if "fasta_records" in st.session_state:
                    del st.session_state["fasta_records"]
                empty_cta(
                    "Nenhum FASTA carregado",
                    "Envie arquivo(s) ou cole um multi-FASTA, escolha a membrana e clique em Predizer todas.",
                )

        if run_fasta_batch:
            batch_recs = st.session_state.get("fasta_records") or []
            if not batch_recs:
                st.warning("Carregue ou cole um FASTA antes de predizer em lote.")
            else:
                charge_note = (
                    f"carga fixa {batch_charge:g}"
                    if batch_charge is not None
                    else "carga estimada por sequência"
                )
                n_total = len(batch_recs)
                progress = st.progress(0)
                status = st.empty()
                rows = []
                for i, r in enumerate(batch_recs):
                    seq = r["sequence"]
                    hdr = (r.get("header") or f"seq_{i+1}")[:48]
                    status.markdown(f"**{i + 1}/{n_total}** · `{hdr}`")
                    try:
                        res = predictor.predict_pair(
                            seq, target_id_lote, net_charge=batch_charge
                        )
                        prob = float(res["pred_high_activity_prob"])
                        _, band_title, _ = activity_band(prob)
                        rows.append(
                            {
                                "header": (r.get("header") or "")[:60],
                                "sequence": seq,
                                "length": len(seq),
                                "origem": r.get("source", ""),
                                "pmi": round(float(res["pmi"]), 3),
                                "prob_calibrada": round(prob, 4),
                                "faixa": band_title,
                                "q_peptide": round(float(res["q_peptide"]), 2),
                                "no_banco": "sim" if lookup_sequence(seq) else "não",
                            }
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "header": (r.get("header") or "")[:60],
                                "sequence": seq,
                                "length": len(seq),
                                "origem": r.get("source", ""),
                                "pmi": None,
                                "prob_calibrada": None,
                                "faixa": "erro",
                                "q_peptide": None,
                                "no_banco": "erro",
                                "erro": str(e),
                            }
                        )
                    progress.progress((i + 1) / n_total)
                progress.empty()
                status.empty()

                batch_df = pd.DataFrame(rows)
                if "prob_calibrada" in batch_df.columns:
                    batch_df = batch_df.sort_values(
                        "prob_calibrada", ascending=False, na_position="last"
                    )
                st.session_state["last_batch"] = {
                    "target_id": target_id_lote,
                    "target_label": format_target_label(target_id_lote),
                    "rows": batch_df.to_dict(orient="records"),
                    "charge_note": charge_note,
                }
                st.session_state.pop("last_batch_narrative", None)

        if st.session_state.get("last_batch"):
            prev = st.session_state["last_batch"]
            with st.container(border=True):
                tile_title(
                    f"Lote × {prev['target_label']}",
                    f"{len(prev['rows'])} peptídeos · {prev.get('charge_note', '')}",
                )
                bdf = pd.DataFrame(prev["rows"])
                n_ok = int(bdf["prob_calibrada"].notna().sum()) if "prob_calibrada" in bdf else 0
                n_high = (
                    int((bdf["prob_calibrada"] >= 0.70).sum())
                    if "prob_calibrada" in bdf
                    else 0
                )
                kpi_row(
                    [
                        {
                            "label": "Preditas",
                            "value": str(n_ok),
                            "hint": f"de {len(prev['rows'])}",
                            "tone": "membrane",
                        },
                        {
                            "label": "≥ 70% (forte)",
                            "value": str(n_high),
                            "hint": "priorizar in vitro",
                            "tone": "ok" if n_high else None,
                        },
                    ],
                    cols=2,
                )
                show_table(bdf, max_text_len=36, highlight_col="prob_calibrada")
                st.download_button(
                    "Baixar CSV do lote",
                    data=bdf.to_csv(index=False).encode("utf-8"),
                    file_name="pepmem_fasta_predicoes.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="batch_csv_persist",
                )

            narrative_engine_caption()
            if st.button("Explicar lote em português", key="btn_narrate_batch"):
                with st.spinner("Gerando explicação…"):
                    nb = st.session_state["last_batch"]
                    out = narrate_batch(
                        target_label=nb["target_label"],
                        rows=nb["rows"],
                        prefer_llm=True,
                    )
                    st.session_state["last_batch_narrative"] = out
            if st.session_state.get("last_batch_narrative"):
                out = st.session_state["last_batch_narrative"]
                render_narrative_box(out["text"], out["source"])
            nb = st.session_state["last_batch"]
            narr = (st.session_state.get("last_batch_narrative") or {}).get("text")
            report = build_batch_report(
                target_label=nb["target_label"],
                rows=nb["rows"],
                narrative=narr,
                charge_note=nb.get("charge_note") or "",
            )
            render_report_downloads(report, "batch_report", "pepmem_relatorio_lote")

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
elif page == "Ranking":
    info_box(
        "Ranking — o que faz",
        "Compara o <strong>mesmo peptídeo</strong> em várias membranas e ordena por "
        "<em>final_score</em> (atividade − toxicidade estimada + bônus de seletividade).<br/>"
        "Topo da lista = alvos mais promissórios para testar primeiro.",
    )

    r1, r2 = st.columns([1.2, 1.0], gap="medium")
    with r1:
        with st.container(border=True):
            tile_title("Parâmetros do ranking", "Score = prob − λ·tox + bônus PMI_sel")
            seq_rank = st.text_input(
                "Sequência para ranking",
                value=st.session_state.get("seq_main", "FFSLIPKLVKGLISAFK"),
                key="rank_seq",
                on_change=force_uppercase,
                args=("rank_seq",),
            )
            type_filter = st.multiselect(
                "Filtrar tipo de alvo",
                options=sorted(targets["target_type"].dropna().unique().tolist()),
                default=[],
            )
            selected = st.multiselect(
                "Alvos específicos (vazio = todos / filtro)",
                options=list(target_options.keys()),
                format_func=lambda x: truncate_text(target_options[x], 40),
            )
            lam = st.slider("Penalização toxicidade (λ)", 0.0, 1.0, 0.5, 0.05)
            run_rank = st.button("Gerar ranking", type="primary")

    with r2:
        with st.container(border=True):
            tile_title("Como ler o score", "Priorização para bancada")
            st.markdown(
                "- **Prob. alta atividade** — chance de alta atividade no alvo\n"
                "- **λ** — quanto penalizar atividade em célula normal (toxicidade proxy)\n"
                "- **PMI seletivo** — PMI no alvo − PMI na célula normal\n"
                "- **Score final** alto → priorize; baixe λ se quiser menos “cautela” toxicológica"
            )

    if run_rank:
        tids = selected or None
        df = predictor.rank_peptide(seq_rank, target_ids=tids, lambda_tox=lam)
        if type_filter:
            df = df[df["target_type"].isin(type_filter)]
        st.session_state["last_rank"] = {
            "sequence": seq_rank,
            "lambda_tox": float(lam),
            "type_filter": list(type_filter) if type_filter else [],
            "rows": df.to_dict(orient="records"),
        }
        st.session_state.pop("last_rank_narrative", None)

    if st.session_state.get("last_rank"):
        snap = st.session_state["last_rank"]
        df = pd.DataFrame(snap["rows"])
        show = df[
            [
                "target_id",
                "target_type",
                "pmi",
                "pmi_sel",
                "pred_high_activity_prob",
                "final_score",
            ]
        ].copy()
        with st.container(border=True):
            tile_title("Matriz de ranking", "Ordenado por score final · verde ≥ 70%")
            show_table(show, max_text_len=32, highlight_col="pred_high_activity_prob")

        if "final_score" in df.columns and "target_id" in df.columns:
            chart = df.set_index("target_id")["final_score"].dropna().sort_values(ascending=False)
            chart.index = chart.index.map(lambda x: truncate_text(str(x), 28))
            with st.container(border=True):
                tile_title("Score final por alvo", "Visual de barras")
                st.bar_chart(chart, color=PM_PURPLE)

        with st.container(border=True):
            tile_title("Explicação em português", "Priorização multi-alvo")
            narrative_engine_caption()
            if st.button("Explicar ranking", type="primary", key="btn_narrate_rank"):
                with st.spinner("Gerando explicação…"):
                    out = narrate_ranking(
                        sequence=snap["sequence"],
                        lambda_tox=float(snap["lambda_tox"]),
                        rows=snap["rows"],
                        prefer_llm=True,
                    )
                    st.session_state["last_rank_narrative"] = out
            if st.session_state.get("last_rank_narrative"):
                out = st.session_state["last_rank_narrative"]
                render_narrative_box(out["text"], out["source"])
            narr = (st.session_state.get("last_rank_narrative") or {}).get("text")
            report = build_ranking_report(
                sequence=snap["sequence"],
                lambda_tox=float(snap["lambda_tox"]),
                rows=snap["rows"],
                narrative=narr,
                type_filter=snap.get("type_filter") or [],
            )
            render_report_downloads(report, "rank_report", "pepmem_relatorio_ranking")
    else:
        empty_cta(
            "Ainda sem ranking",
            "Cole uma sequência à esquerda, ajuste λ se quiser, e clique em Gerar ranking.",
        )

    st.markdown("---")
    with st.container(border=True):
        tile_title("Ranking pré-calculado do projeto", "Baseline offline")
        ranking_path = ROOT / "data" / "processed" / "models" / "project_ranking_baseline.csv"
        if ranking_path.exists():
            pre = pd.read_csv(ranking_path)
            filt = st.selectbox(
                "Filtrar alvo",
                ["Todos"] + sorted(pre["target_id"].unique().tolist()),
                key="pre_filt",
            )
            view = pre if filt == "Todos" else pre[pre["target_id"] == filt]
            show_table(
                view.nlargest(15, "pred_high_activity_prob")[
                    ["peptide_id", "target_id", "pmi", "pmi_sel", "pred_high_activity_prob"]
                ],
                max_text_len=28,
                highlight_col="pred_high_activity_prob",
            )

# ---------------------------------------------------------------------------
# XAI (lazy: só roda quando esta página está ativa)
# ---------------------------------------------------------------------------
elif page == "XAI":
    with st.container(border=True):
        tile_title("Como funciona o SHAP neste projeto", "XAI do Random Forest · PepMem-AI")
        st.markdown(
            """
**SHAP** estima **quanto cada descritor contribui** para a chance de **alta atividade**
(MIC ≤ 3,4 µM), via **TreeExplainer** no Random Forest.

| Sinal | Significado |
|-------|-------------|
| **Positivo** | Empurra **para** alta atividade |
| **Negativo** | Empurra **contra** alta atividade |
"""
        )
        with st.expander("Saiba mais — SHAP, ESM-2 e limites"):
            st.markdown(
                f"""
A ideia vem dos valores de Shapley: cada feature recebe um “crédito” justo pela diferença
entre a predição com e sem ela. No PepMem-AI usamos TreeExplainer (baseline e, no Space, multimodal).

### Três visuais
1. **Importância global** — média de |SHAP| nos ~{n_train} pares MIC
2. **Beeswarm** — cada ponto = um par MIC; cor ≈ valor do descritor
3. **SHAP local** — barras para **uma** sequência × membrana

### ESM-2
Modelo de linguagem de proteínas (`facebook/esm2_t6_8M_UR50D`). No multimodal, o embedding
(~320 dims, mean-pool) entra com as features clássicas; no SHAP aparece agregado como
**“ESM-2 (embedding agregado)”**. No Cloud leve (sem PyTorch) o app usa só o baseline.

**Por quê t6 8M:** feito para proteínas, só precisa da sequência, complementa o PMI e cabe
em CPU/Space — melhor custo–benefício para o PoC.

### Limites
- SHAP explica o **modelo**, não prova mecanismo biológico nem substitui MIC.
- Com poucos análogos, importância global pode refletir vieses da amostra.
- Probabilidade do dashboard é **calibrada (LOPO)**; SHAP local fala da saída do RF.

Treino atual: **{n_train}** pares MIC · rótulo: MIC ≤ 3,4 µM = alta atividade.
"""
            )

    global_report = predictor.global_shap_report()
    baseline_path = ROOT / "data" / "processed" / "models" / "shap_global_baseline.json"
    baseline_report = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    )

    with st.container(border=True):
        tile_title("Explicação em português", "Panorama SHAP global")
        narrative_engine_caption()
        if st.button("Explicar panorama SHAP", type="primary", key="btn_narrate_shap_global"):
            with st.spinner("Gerando explicação…"):
                out = narrate_shap_overview(
                    n_train=n_train,
                    baseline_importance=(baseline_report or {}).get("global_importance"),
                    multimodal_importance=(global_report or {}).get("global_importance"),
                    prefer_llm=True,
                )
                st.session_state["last_shap_overview_narrative"] = out
        if st.session_state.get("last_shap_overview_narrative"):
            out = st.session_state["last_shap_overview_narrative"]
            render_narrative_box(out["text"], out["source"])
        narr = (st.session_state.get("last_shap_overview_narrative") or {}).get("text")
        report = build_shap_overview_report(
            n_train=n_train,
            narrative=narr,
            baseline_importance=(baseline_report or {}).get("global_importance"),
            multimodal_importance=(global_report or {}).get("global_importance"),
        )
        render_report_downloads(report, "shap_overview_report", "pepmem_relatorio_shap")

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            tile_title(
                "Importância global — multimodal",
                f"{(global_report or {}).get('n_samples', '—')} MICs",
            )
            if global_report:
                st.pyplot(
                    plot_global_importance(global_report["global_importance"]),
                    clear_figure=True,
                    use_container_width=True,
                )
    with g2:
        with st.container(border=True):
            tile_title(
                "Importância global — baseline",
                f"{(baseline_report or {}).get('n_samples', '—')} MICs",
            )
            if baseline_report:
                st.pyplot(
                    plot_global_importance(
                        baseline_report["global_importance"],
                        title="Importância |SHAP| — baseline",
                    ),
                    clear_figure=True,
                    use_container_width=True,
                )

    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            tile_title("Beeswarm multimodal", "Cada ponto = um par MIC")
            try:
                show_png(cached_beeswarm(True, n_train))
            except Exception as e:
                st.error(f"Beeswarm multimodal: {e}")
    with b2:
        with st.container(border=True):
            tile_title("Beeswarm baseline", "Descritores clássicos + PMI")
            try:
                show_png(cached_beeswarm(False, n_train))
            except Exception as e:
                st.error(f"Beeswarm baseline: {e}")

    with st.container(border=True):
        tile_title("Explicação local", "Instância peptídeo × membrana")
        if not st.session_state.get("last_xai_local"):
            empty_cta(
                "Ainda sem SHAP local",
                "Informe a sequência e a membrana abaixo e clique em Calcular SHAP local.",
            )
        xai_seq = st.text_input(
            "Sequência",
            value=st.session_state.get("seq_main", "FFSLIPSLVGGLISAFK"),
            key="xai_seq",
            on_change=force_uppercase,
            args=("xai_seq",),
        )
        xc1, xc2 = st.columns(2)
        with xc1:
            xai_use_charge = st.checkbox("Informar carga", value=True, key="xai_use_charge")
            xai_charge = st.number_input("Carga líquida", value=3.0, step=1.0, key="xai_charge")
        with xc2:
            xai_target = st.selectbox(
                "Membrana-alvo",
                options=list(target_options.keys()),
                format_func=format_target_label,
                key="xai_target",
                index=list(target_options.keys()).index("S_aureus_ATCC29213")
                if "S_aureus_ATCC29213" in target_options
                else 0,
            )
        charge_xai = float(xai_charge) if xai_use_charge else None
        if st.button("Calcular SHAP local", type="primary", key="xai_btn"):
            try:
                expl = cached_explain(xai_seq, xai_target, charge_xai)
                st.session_state["last_xai_local"] = {
                    "expl": expl,
                    "sequence": xai_seq,
                    "target_id": xai_target,
                    "target_label": target_options[xai_target],
                    "charge": charge_xai,
                    "hit": lookup_sequence(xai_seq),
                }
                st.session_state.pop("last_xai_narrative", None)
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("last_xai_local"):
            snap = st.session_state["last_xai_local"]
            expl = snap["expl"]
            render_result_banner(float(expl["pred_high_activity_prob"]), snap.get("hit"))
            render_shap_block(expl, snap["target_label"])

            st.markdown("---")
            tile_title("Explicação em português", "SHAP local desta instância")
            narrative_engine_caption()
            if st.button("Explicar SHAP local", key="btn_narrate_shap_local"):
                with st.spinner("Gerando explicação…"):
                    out = narrate_single(
                        sequence=snap["sequence"],
                        target_label=snap["target_label"],
                        prob=float(expl["pred_high_activity_prob"]),
                        pmi=float(expl["pmi"]) if expl.get("pmi") is not None else None,
                        q_peptide=float(expl["q_peptide"]) if expl.get("q_peptide") is not None else None,
                        shap_top=expl.get("shap_contributions"),
                        in_project=snap.get("hit") is not None,
                        prefer_llm=True,
                    )
                    st.session_state["last_xai_narrative"] = out
            if st.session_state.get("last_xai_narrative"):
                out = st.session_state["last_xai_narrative"]
                render_narrative_box(out["text"], out["source"])
            narr = (st.session_state.get("last_xai_narrative") or {}).get("text")
            report = build_single_report(
                sequence=snap["sequence"],
                target_label=snap["target_label"],
                res=expl,
                narrative=narr,
                shap_top=expl.get("shap_contributions"),
                in_project=snap.get("hit") is not None,
            )
            render_report_downloads(report, "xai_local_report", "pepmem_relatorio_shap_local")

# ---------------------------------------------------------------------------
# Dados (+ API no expander)
# ---------------------------------------------------------------------------
elif page == "Dados":
    info_box(
        "Datasets",
        "Resumo do que alimenta o modelo: peptídeos do projeto, pares e MICs. "
        "Para incluir resultado novo da bancada, use <code>data/bench/</code> "
        "e o script de importação.",
    )

    summary_path = ROOT / "data" / "processed" / "build_summary.json"
    bench_report = ROOT / "data" / "bench" / "import_report.json"
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")

    kpi_row(
        [
            {"label": "Peptídeos (base)", "value": f"{len(base):,}", "tone": "membrane"},
            {"label": "Projeto", "value": str(len(project_df) if not project_df.empty else "—")},
            {"label": "Pares", "value": f"{len(pairs):,}"},
            {
                "label": "MICs",
                "value": str(int(pairs["mic_value"].notna().sum())),
                "tone": "ok",
            },
        ],
        cols=4,
    )

    if bench_report.exists():
        with st.expander("Relatório bancada"):
            st.json(json.loads(bench_report.read_text(encoding="utf-8")))
    if summary_path.exists():
        with st.expander("build_summary.json"):
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))

    with st.container(border=True):
        tile_title("Peptídeos do projeto", "Stigmurin, StigA, TsAP-2 e análogos")
        st.caption(
            "P01–P09 podem ter sequência placeholder — preferir P10–P18 para testes."
        )
        if not project_df.empty:
            view = project_df[["peptide_id", "name", "sequence", "net_charge", "source"]].copy()
            native_seqs = set(
                project_df.loc[project_df["peptide_id"].isin(["P05", "P10"]), "sequence"].astype(str)
            )

            def flag(row):
                pid = str(row["peptide_id"])
                seq = str(row["sequence"])
                if pid in {"P01", "P02", "P03", "P04", "P06", "P07", "P08", "P09"} and seq in native_seqs:
                    return "placeholder?"
                if pid in {"P10", "P11", "P12", "P13", "P14", "P15", "P16", "P05", "P17", "P18"}:
                    return "no banco / MIC"
                return ""

            view["nota"] = view.apply(flag, axis=1)
            show_table(view, max_text_len=40)

    with st.expander("API local (FastAPI) — integração HTTP"):
        st.markdown(
            """
Endpoints (`uvicorn api.main:app --port 8001`):

| Método | Rota | Uso |
|--------|------|-----|
| GET | `/health` | Status |
| GET | `/targets` | Membranas |
| POST | `/predict` | Um par |
| POST | `/explain` | SHAP local |
| POST | `/rank` | Ranking |

```bash
curl -X POST http://localhost:8001/predict \\
  -H "Content-Type: application/json" \\
  -d '{"sequence":"FFSLIPKLVAGLISAFK","target_id":"S_aureus_ATCC29213","net_charge":3}'
```
"""
        )
        if info:
            st.caption(
                f"Modelo carregado: {info.get('model_type', '—')} · "
                f"amostras {info.get('n_samples', '—')} · "
                f"LOPO AUC {info.get('leave_one_peptide_auc', info.get('loo_auc', '—'))}"
            )
