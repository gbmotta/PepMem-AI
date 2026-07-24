"""Dashboard Streamlit do PepMem-AI — layout estilo Power BI.

Relatório analítico: barra de relatório · painel de filtros · páginas (abas)
· faixa de KPIs · tiles de visuais. Paleta peçonha / peptídeo / membrana.

Execução local:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

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
    path = ROOT / "data" / "processed" / "pepmem_base_project.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
def cached_explain(sequence: str, target_id: str, net_charge: float | None) -> dict:
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


def truncate_text(value: object, max_len: int = 40) -> object:
    """Trunca strings longas em tabelas densas."""
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def show_table(df: pd.DataFrame, max_text_len: int = 40) -> None:
    """Tabela truncada em estilo relatório."""
    view = df.copy()
    for col in view.columns:
        if view[col].dtype == object:
            view[col] = view[col].map(lambda x: truncate_text(x, max_text_len))
    st.dataframe(view, use_container_width=True, hide_index=True)


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
    """Banner de interpretação + badge no banco / fora do treino."""
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
        f'<div class="pm-result {css}">{badge}<h3>{html.escape(title)}</h3>'
        f"<p>{html.escape(msg)}</p></div>",
        unsafe_allow_html=True,
    )


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
    with st.expander("Tabela de contribuições"):
        df = pd.DataFrame(expl["shap_contributions"])[["label", "shap_value", "group"]]
        df["shap_value"] = df["shap_value"].map(lambda x: f"{x:+.4f}")
        show_table(df, max_text_len=36)
    st.caption("Valores positivos → favorecem alta atividade · negativos → desfavorecem")


def apply_preset(seq: str, charge: float) -> None:
    """Callback on_click — define sessão antes dos widgets no próximo rerun."""
    st.session_state["seq_main"] = seq
    st.session_state["charge_main"] = float(charge)
    st.session_state["use_charge_main"] = True


def apply_fasta_sequence(seq: str, header: str = "") -> None:
    """Callback: preenche a sequência a partir de um registro FASTA."""
    st.session_state["seq_main"] = seq
    if header:
        st.session_state["fasta_header"] = header


def load_fasta_records(uploaded) -> list[dict]:
    """Lê upload Streamlit (.fasta/.fa/.txt) e devolve registros parseados."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from peptide_utils import parse_fasta_text

    raw = uploaded.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    records = parse_fasta_text(text)
    return [r for r in records if r.get("sequence")]


# Defaults de sessão (antes dos widgets)
if "seq_main" not in st.session_state:
    st.session_state["seq_main"] = "FFSLIPKLVKGLISAFK"
if "charge_main" not in st.session_state:
    st.session_state["charge_main"] = 3.0
if "use_charge_main" not in st.session_state:
    st.session_state["use_charge_main"] = True

# --- sidebar = painel de filtros ---
with st.sidebar:
    st.markdown("### Modelo")
    st.caption("InovAI Lab · UFRN")
    st.write(info.get("model_type", "Random Forest + PMI"))
    st.write(f"Treino: **{n_train}** MICs")
    if lope is not None:
        st.write(f"AUC (peptídeo deixado de fora): **{float(lope):.3f}**")
    st.caption("Alta atividade = MIC ≤ 3,4 µM")

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

tab_pred, tab_rank, tab_xai, tab_data, tab_api = st.tabs(
    ["Predição", "Ranking", "XAI (SHAP)", "Datasets", "API"]
)

# --- aba Predição ---
with tab_pred:
    info_box(
        "Como ler a predição",
        "<strong>≥ 70%</strong> → candidato forte · "
        "<strong>40–70%</strong> → intermediário · "
        "<strong>&lt; 40%</strong> → baixa chance.<br/>"
        "<strong>PMI</strong> alto sugere boa interação eletrostática/hidrofóbica. "
        "O <strong>intervalo</strong> mostra incerteza entre as árvores do modelo. "
        "Confirme sempre na bancada.",
    )

    with st.container(border=True):
        tile_title("Par peptídeo × membrana", "Cole a sequência, envie FASTA ou escolha um exemplo")
        st.caption(
            "Exemplos **conhecidos do projeto** já têm MIC no treino. "
            "Exemplos **novos** servem só para testar a predição (não estão no modelo)."
        )

        preset_labels = [p[0] for p in PRESETS]
        chosen = st.selectbox("Carregar exemplo (opcional)", options=preset_labels, key="preset_select")
        if chosen != preset_labels[0]:
            _, pseq, pch = next(p for p in PRESETS if p[0] == chosen)
            if st.button("Aplicar exemplo", use_container_width=True):
                apply_preset(pseq, float(pch))
                st.rerun()

        st.markdown(
            '<div class="pm-hint-box">'
            "<strong>Peptídeo novo com MIC da bancada?</strong> "
            "Edite <code>data/bench/mic_bench.csv</code> e rode "
            "<code>python scripts/import_bench_mic.py --retrain</code>."
            "</div>",
            unsafe_allow_html=True,
        )

        fasta_file = st.file_uploader(
            "Arquivo FASTA (opcional)",
            type=["fasta", "fa", "faa", "fna", "txt"],
            help="Um ou vários peptídeos no formato >header + sequência",
            key="fasta_upload",
        )
        run_fasta_batch = False
        if fasta_file is not None:
            try:
                fasta_recs = load_fasta_records(fasta_file)
            except Exception as e:
                st.error(f"FASTA inválido: {e}")
                fasta_recs = []
            if not fasta_recs:
                st.warning("Nenhuma sequência válida encontrada no FASTA.")
            else:
                st.caption(f"**{len(fasta_recs)}** sequência(s) no arquivo.")
                labels = [
                    f"{(r.get('header') or f'seq_{i+1}')[:48]} · {r['sequence'][:12]}…"
                    if len(r["sequence"]) > 12
                    else f"{(r.get('header') or f'seq_{i+1}')[:48]} · {r['sequence']}"
                    for i, r in enumerate(fasta_recs)
                ]
                pick = st.selectbox(
                    "Escolher sequência do FASTA",
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
                        "Predizer todas do FASTA",
                        use_container_width=True,
                        key="fasta_batch_btn",
                    )
                st.session_state["fasta_records"] = fasta_recs
        elif "fasta_records" in st.session_state:
            del st.session_state["fasta_records"]

        sequence = st.text_input(
            "Sequência (letra única)",
            key="seq_main",
            help="Cole a sequência de aminoácidos",
            placeholder="Ex.: FFSLIPKLVAGLISAFK",
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

        preferred = "S_aureus_ATCC29213"
        keys = list(target_options.keys())
        idx = keys.index(preferred) if preferred in keys else 0
        target_id = st.selectbox(
            "Membrana-alvo",
            options=keys,
            index=idx,
            format_func=format_target_label,
        )
        run_pred = st.button("Predizer", type="primary", use_container_width=True)

    if run_fasta_batch:
        batch_recs = st.session_state.get("fasta_records") or []
        if not batch_recs:
            st.warning("Carregue um FASTA antes de predizer em lote.")
        else:
            with st.spinner(f"Predizendo {len(batch_recs)} sequências…"):
                rows = []
                for r in batch_recs:
                    seq = r["sequence"]
                    try:
                        res = predictor.predict_pair(seq, target_id, net_charge=charge)
                        rows.append(
                            {
                                "header": (r.get("header") or "")[:60],
                                "sequence": seq,
                                "length": len(seq),
                                "pmi": round(float(res["pmi"]), 3),
                                "prob_calibrada": round(float(res["pred_high_activity_prob"]), 4),
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
                                "pmi": None,
                                "prob_calibrada": None,
                                "q_peptide": None,
                                "no_banco": "erro",
                                "erro": str(e),
                            }
                        )
            with st.container(border=True):
                tile_title(
                    f"Lote FASTA × {format_target_label(target_id)}",
                    f"{len(rows)} sequências · mesma membrana e carga",
                )
                batch_df = pd.DataFrame(rows)
                if "prob_calibrada" in batch_df.columns:
                    batch_df = batch_df.sort_values(
                        "prob_calibrada", ascending=False, na_position="last"
                    )
                show_table(batch_df, max_text_len=36)
                st.download_button(
                    "Baixar CSV do lote",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="pepmem_fasta_predicoes.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    if run_pred:
        with st.spinner("Descritores · PMI · modelo RF…"):
            try:
                res = predictor.predict_pair(sequence, target_id, net_charge=charge)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        prob = float(res["pred_high_activity_prob"])
        render_result_banner(prob, hit)

        lo = res.get("pred_interval_low")
        hi = res.get("pred_interval_high")
        interval = (
            f"{100 * float(lo):.0f}–{100 * float(hi):.0f}%"
            if lo is not None and hi is not None
            else "—"
        )
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
            tile_title("Vizinhos no treino", "Identidade + cosine ESM-2")
            st.caption(
                "Peptídeos parecidos no treino ajudam a contextualizar o resultado: "
                "se os vizinhos têm MIC baixo no mesmo alvo, a predição fica mais crível."
            )
            neighbors = predictor.find_neighbors(sequence, k=5, target_id=target_id)
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

        with st.container(border=True):
            tile_title("Explicação SHAP", "Contribuições locais do RF")
            try:
                expl = cached_explain(sequence, target_id, charge)
                render_shap_block(expl, target_options[target_id])
            except Exception as e:
                st.warning(f"SHAP indisponível: {e}")

# --- aba Ranking ---
with tab_rank:
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
                "- **prob** — chance de alta atividade no alvo\n"
                "- **λ** — quanto penalizar atividade em célula normal (toxicidade proxy)\n"
                "- **PMI_sel** — PMI no alvo − PMI na célula normal (seletividade)\n"
                "- **final_score** alto → priorize; baixe λ se quiser menos “cautela” toxicológica"
            )

    if run_rank:
        tids = selected or None
        df = predictor.rank_peptide(seq_rank, target_ids=tids, lambda_tox=lam)
        if type_filter:
            df = df[df["target_type"].isin(type_filter)]
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
        show["pred_high_activity_prob"] = show["pred_high_activity_prob"].map(lambda x: f"{x:.1%}")
        with st.container(border=True):
            tile_title("Matriz de ranking", "Ordenado por final_score")
            show_table(show, max_text_len=32)

        chart = df.set_index("target_id")["final_score"].dropna().sort_values(ascending=False)
        chart.index = chart.index.map(lambda x: truncate_text(str(x), 28))
        with st.container(border=True):
            tile_title("Score final por alvo", "Visual de barras")
            st.bar_chart(chart, color=PM_PURPLE)

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
            )

# --- aba XAI ---
with tab_xai:
    info_box(
        "SHAP — interpretação rápida",
        "Barras/valores <strong>positivos</strong> empurram para alta atividade; "
        "<strong>negativos</strong> empurram para o contrário.<br/>"
        "O beeswarm mostra o efeito de cada descritor em todo o treino. "
        "É uma explicação do modelo, não prova biológica.",
    )
    st.caption(
        f"Treino atual: **{n_train}** pares MIC · rótulo: MIC ≤ 3,4 µM = alta atividade."
    )

    global_report = predictor.global_shap_report()
    baseline_path = ROOT / "data" / "processed" / "models" / "shap_global_baseline.json"
    baseline_report = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    )

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            tile_title("Importância global — multimodal", f"{(global_report or {}).get('n_samples', '—')} MICs")
            if global_report:
                st.pyplot(
                    plot_global_importance(global_report["global_importance"]),
                    clear_figure=True,
                    use_container_width=True,
                )
    with g2:
        with st.container(border=True):
            tile_title("Importância global — baseline", f"{(baseline_report or {}).get('n_samples', '—')} MICs")
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
                st.image(cached_beeswarm(True, n_train), use_container_width=True)
            except Exception as e:
                st.error(f"Beeswarm multimodal: {e}")
    with b2:
        with st.container(border=True):
            tile_title("Beeswarm baseline", "Descritores clássicos + PMI")
            try:
                st.image(cached_beeswarm(False, n_train), use_container_width=True)
            except Exception as e:
                st.error(f"Beeswarm baseline: {e}")

    with st.container(border=True):
        tile_title("Explicação local", "Instância peptídeo × membrana")
        xai_seq = st.text_input(
            "Sequência",
            value=st.session_state.get("seq_main", "FFSLIPSLVGGLISAFK"),
            key="xai_seq",
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
                render_result_banner(float(expl["pred_high_activity_prob"]), lookup_sequence(xai_seq))
                render_shap_block(expl, target_options[xai_target])
            except Exception as e:
                st.error(str(e))

# --- aba Datasets ---
with tab_data:
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

# --- aba API ---
with tab_api:
    info_box(
        "API",
        "Mesmas predições via HTTP (útil para integrar com outros sistemas). "
        "No Cloud público o foco é o dashboard; a API roda localmente com FastAPI.",
    )

    with st.container(border=True):
        tile_title("API local (FastAPI)", "Integração via HTTP")
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
        with st.container(border=True):
            tile_title("Modelo carregado", "Metadados do relatório")
            st.write(f"- Tipo: {info.get('model_type', '—')}")
            st.write(f"- Amostras: {info.get('n_samples', '—')}")
            st.write(f"- LOO AUC: {info.get('loo_auc', '—')}")
            st.write(f"- Leave-peptide AUC: {info.get('leave_one_peptide_auc', '—')}")
