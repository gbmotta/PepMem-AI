"""PepMem-AI Streamlit dashboard."""

from __future__ import annotations

import json
import sys

import pandas as pd
import streamlit as st

from pepmem.paths import project_root

ROOT = project_root()
sys.path.insert(0, str(ROOT))

from pepmem.predictor import PepMemPredictor
from pepmem.shap_explain import plot_beeswarm, plot_contributions, plot_global_importance

# --- page ---
st.set_page_config(
    page_title="PepMem-AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme: bicamada + peçonha — versão suave (menos saturação)
st.markdown(
    """
    <style>
      :root {
        --pm-amber: #b0893a;
        --pm-amber-deep: #8a6b2e;
        --pm-amber-soft: #f2ead8;
        --pm-navy: #1a2433;
        --pm-membrane: #243447;
        --pm-polar: #5f7d8a;
        --pm-aqueous: #f3f5f7;
        --pm-ink: #1e2733;
        --pm-muted: #6b7785;
        --pm-ok: #3d6b5c;
        --pm-mid: #9a6b2f;
        --pm-low: #8b4a4a;
      }
      html, body, .stApp {
        /* Bicamada bem diluída — só um véu, sem faixas fortes */
        background:
          linear-gradient(180deg,
            #f5f7f9 0%,
            #eef2f5 35%,
            #f4efe6 48%,
            #eef2f5 62%,
            #f5f7f9 100%
          );
        background-attachment: fixed;
        color: var(--pm-ink);
      }
      .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2433 0%, #243447 100%);
        border-right: 1px solid #2f4054;
      }
      [data-testid="stSidebar"] * { color: #e8edf2 !important; }
      [data-testid="stSidebar"] .stCaption,
      [data-testid="stSidebar"] label {
        color: #9aa8b5 !important;
      }
      [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #2a3a4d !important;
        border-color: #3d5166 !important;
      }
      [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #3d5166;
        color: #e4d8bc !important;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #b0893a;
        background: rgba(176, 137, 58, 0.1);
      }
      h1.pm-brand {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        letter-spacing: -0.03em;
        font-size: 2.2rem;
        margin: 0 0 0.15rem 0;
        color: var(--pm-navy);
      }
      h1.pm-brand span.pm-accent {
        color: var(--pm-amber-deep);
      }
      p.pm-tagline {
        color: var(--pm-muted);
        font-size: 1.02rem;
        margin: 0 0 1.1rem 0;
        max-width: 42rem;
      }
      .pm-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.35rem 0 0.85rem; }
      .pm-chip {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid #c5d0da;
        background: rgba(255,255,255,0.7);
        color: #3a4a5a;
      }
      .pm-chip.novel {
        border-color: #d4c4a0;
        background: var(--pm-amber-soft);
        color: var(--pm-amber-deep);
      }
      .pm-result {
        border-radius: 12px;
        padding: 0.95rem 1.05rem;
        margin: 0.7rem 0 0.95rem;
        border: 1px solid #d0d8e0;
        background: rgba(255,255,255,0.88);
      }
      .pm-result.high { border-left: 4px solid var(--pm-amber); }
      .pm-result.mid  { border-left: 4px solid var(--pm-mid); }
      .pm-result.low  { border-left: 4px solid var(--pm-low); }
      .pm-result h3 {
        margin: 0 0 0.3rem 0;
        font-size: 1.02rem;
        color: var(--pm-ink);
      }
      .pm-result p { margin: 0; color: var(--pm-muted); font-size: 0.9rem; line-height: 1.45; }
      .pm-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.18rem 0.45rem;
        border-radius: 5px;
        margin-bottom: 0.4rem;
      }
      .pm-badge.in { background: #e4ebf2; color: #2a3d52; }
      .pm-badge.out { background: var(--pm-amber-soft); color: var(--pm-amber-deep); }
      .pm-badge.ph { background: #e8ebef; color: #5a6570; }
      [data-testid="stMetric"] {
        background: rgba(255,255,255,0.9);
        border: 1px solid #d5dde5;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 1px solid #d5dde5;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
      }
      div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
      [data-testid="stImage"] img,
      [data-testid="stPyplot"] img { border-radius: 8px; }
      .stButton > button[kind="primary"],
      button[data-testid="baseButton-primary"] {
        background: var(--pm-amber-deep) !important;
        border: 1px solid var(--pm-amber-deep) !important;
        color: #faf8f4 !important;
        font-weight: 600 !important;
      }
      .stButton > button[kind="primary"]:hover,
      button[data-testid="baseButton-primary"]:hover {
        background: var(--pm-amber) !important;
        border-color: var(--pm-amber) !important;
      }
      .stMarkdown table { width: 100%; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- presets ---
PRESETS_KNOWN = [
    ("StigA6 (P11)", "FFSLIPKLVKGLISAFK", 3.0),
    ("Stigmurin (P10)", "FFSLIPSLVGGLISAFK", 3.0),
    ("StigA16 (P12)", "FFKLIPKLVKGLISAFK", 4.0),
    ("TsAP-2-A16 (P17)", "FLRMIPGLIRGLIRAFR", 5.0),
    ("StigA31 (P16)", "FFKLIPKLVKKLIKAFK", 7.0),
]
PRESETS_NOVEL = [
    ("Mutante S→A (fora do treino)", "FFSLIPKLVAGLISAFK", 3.0),
    ("+carga estilo A16", "FFKLIPKLVAGLISAFK", 4.0),
    ("Variante hidrofóbica", "FFALIPKLVKGLISAFK", 3.0),
]


@st.cache_resource
def get_predictor() -> PepMemPredictor:
    return PepMemPredictor(use_embeddings=True)


@st.cache_data
def load_project_peptides() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "pepmem_base_project.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Gerando beeswarm SHAP...")
def cached_beeswarm(use_embeddings: bool, n_mic: int, _layout_version: int = 5) -> bytes:
    import io

    import joblib
    import matplotlib.pyplot as plt

    fname = "multimodal_mic_rf.joblib" if use_embeddings else "baseline_mic_rf.joblib"
    path = ROOT / "data" / "processed" / "models" / fname
    if not path.exists():
        raise FileNotFoundError(f"Modelo ausente: {fname}")
    pipe = joblib.load(path)
    kind = "multimodal" if use_embeddings else "baseline"
    fig = plot_beeswarm(pipe, use_embeddings, title=f"Beeswarm SHAP — {kind} ({n_mic} MICs)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="Calculando SHAP...")
def cached_explain(sequence: str, target_id: str, net_charge: float | None) -> dict:
    return get_predictor().explain_pair(sequence, target_id, net_charge=net_charge)


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


def format_target_label(target_id: str) -> str:
    name = target_options.get(target_id, target_id)
    short = name if len(name) <= 40 else f"{name[:37]}…"
    return f"{short}"


def truncate_text(value: object, max_len: int = 40) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def show_table(df: pd.DataFrame, max_text_len: int = 40) -> None:
    view = df.copy()
    for col in view.columns:
        if view[col].dtype == object:
            view[col] = view[col].map(lambda x: truncate_text(x, max_text_len))
    st.table(view)


def activity_band(prob: float) -> tuple[str, str, str]:
    """Return (css_class, badge_text, interpretation)."""
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
    key = "".join(c for c in seq.upper() if c.isalpha())
    return seq_to_project.get(key)


def render_result_banner(prob: float, in_db: dict | None) -> None:
    css, title, msg = activity_band(prob)
    if in_db is not None:
        badge = f'<span class="pm-badge in">No banco · {in_db.get("peptide_id")} · {in_db.get("name")}</span>'
    else:
        badge = '<span class="pm-badge out">Fora do treino · predição generalizada</span>'
    st.markdown(
        f'<div class="pm-result {css}">{badge}<h3>{title}</h3><p>{msg}</p></div>',
        unsafe_allow_html=True,
    )


def render_shap_block(expl: dict, target_label: str) -> None:
    st.metric("Prob. alta atividade (RF)", f"{expl['pred_high_activity_prob']:.1%}")
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
    """Callback on_click — roda antes dos widgets no rerun."""
    st.session_state["seq_main"] = seq
    st.session_state["charge_main"] = float(charge)
    st.session_state["use_charge_main"] = True


# Defaults antes de qualquer widget com essas keys
if "seq_main" not in st.session_state:
    st.session_state["seq_main"] = "FFSLIPKLVKGLISAFK"
if "charge_main" not in st.session_state:
    st.session_state["charge_main"] = 3.0
if "use_charge_main" not in st.session_state:
    st.session_state["use_charge_main"] = True

# --- sidebar ---
with st.sidebar:
    st.markdown("### PepMem-AI")
    st.caption("InovAI Lab · UFRN")
    st.markdown("---")
    st.markdown("**Modelo**")
    st.write(info.get("model_type", "Random Forest + ESM-2"))
    st.write(f"Treino: **{n_train}** MICs")
    if loo_auc is not None:
        st.write(f"LOO amostra AUC: **{float(loo_auc):.3f}**")
    lope = info.get("leave_one_peptide_auc")
    if lope is not None:
        st.write(f"Leave-peptide AUC: **{float(lope):.3f}**")
        st.caption("Leave-peptide = métrica honestada (sem vazar análogos 90% idênticos)")
    st.caption("Rótulo: MIC ≤ 3,4 µM = alta atividade · probs calibradas (isotonic)")
    st.markdown("---")
    st.markdown("**Atalhos de teste**")
    st.caption("No banco (valida UI)")
    for label, seq, ch in PRESETS_KNOWN:
        st.button(
            label,
            key=f"sb_k_{label}",
            use_container_width=True,
            on_click=apply_preset,
            args=(seq, ch),
        )
    st.caption("Fora do banco (generalização)")
    for label, seq, ch in PRESETS_NOVEL:
        st.button(
            label,
            key=f"sb_n_{label}",
            use_container_width=True,
            on_click=apply_preset,
            args=(seq, ch),
        )


# --- header ---
st.markdown('<h1 class="pm-brand">Pep<span class="pm-accent">Mem</span>-AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="pm-tagline">Interação peptídeo–membrana inspirada em peçonhas de '
    "<em>Tityus stigmurus</em> — PMI + Random Forest multimodal para priorizar ensaios in vitro.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="pm-chip-row">'
    f'<span class="pm-chip">{n_train} MICs no treino</span>'
    '<span class="pm-chip">Bicamada · carga · µH</span>'
    '<span class="pm-chip novel">Peptídeo catiônico × membrana aniônica</span>'
    "</div>",
    unsafe_allow_html=True,
)

tab_pred, tab_rank, tab_xai, tab_data, tab_api = st.tabs(
    ["Predição", "Ranking", "XAI (SHAP)", "Datasets", "API"]
)

with tab_pred:
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.subheader("Par peptídeo × membrana")
        sequence = st.text_input(
            "Sequência (letra única)",
            key="seq_main",
            help="Ex.: StigA6 FFSLIPKLVKGLISAFK — ou uma mutação fora do treino",
        )
        hit = lookup_sequence(sequence or "")
        if hit is not None:
            st.caption(f"Correspondência: **{hit.get('peptide_id')} · {hit.get('name')}** (já no projeto)")
        else:
            st.caption("Sequência **não** encontrada no banco do projeto — predição generalizada.")

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

    with right:
        st.subheader("Exemplos rápidos")
        st.markdown("**Já no banco**")
        cols = st.columns(2)
        for i, (label, seq, ch) in enumerate(PRESETS_KNOWN[:4]):
            cols[i % 2].button(
                label,
                key=f"qk_{i}",
                use_container_width=True,
                on_click=apply_preset,
                args=(seq, ch),
            )
        st.markdown("**Fora do banco**")
        for i, (label, seq, ch) in enumerate(PRESETS_NOVEL):
            st.button(
                label,
                key=f"qn_{i}",
                use_container_width=True,
                on_click=apply_preset,
                args=(seq, ch),
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

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PMI", f"{res['pmi']:.3f}")
        lo = res.get("pred_interval_low")
        hi = res.get("pred_interval_high")
        interval = (
            f"{100 * float(lo):.0f}–{100 * float(hi):.0f}%"
            if lo is not None and hi is not None
            else "—"
        )
        m2.metric(
            "Prob. calibrada",
            f"{prob:.1%}",
            help="Isotonic em leave-one-peptide-out",
        )
        m3.metric("Intervalo (árvores RF)", interval)
        m4.metric("Carga (q)", f"{res['q_peptide']:.1f}")

        if res.get("pred_high_activity_prob_raw") is not None:
            st.caption(
                f"Prob. bruta (antes da calibração): {float(res['pred_high_activity_prob_raw']):.1%} · "
                f"σ árvores: {float(res.get('pred_uncertainty_std') or 0):.3f}"
            )

        st.markdown("#### Vizinhos no treino")
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
                f"MIC mediana {top['mic_median_uM']} µM · "
                f"{100 * top['frac_high_activity']:.0f}% dos pares com alta atividade"
            )
        else:
            st.caption("Sem índice MIC para vizinhos.")

        with st.expander("Detalhes da resposta"):
            st.code(res["sequence"], language=None)
            st.json({k: v for k, v in res.items() if k != "sequence"})

        st.markdown("#### Explicação SHAP")
        try:
            expl = cached_explain(sequence, target_id, charge)
            render_shap_block(expl, target_options[target_id])
        except Exception as e:
            st.warning(f"SHAP indisponível: {e}")

with tab_rank:
    r1, r2 = st.columns([1.2, 1.0], gap="large")
    with r1:
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
        st.info(
            "O **final_score** combina probabilidade de atividade, "
            "penalidade em célula normal (λ) e bônus de PMI_sel."
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
        show_table(show, max_text_len=32)

        chart = df.set_index("target_id")["final_score"].dropna().sort_values(ascending=False)
        chart.index = chart.index.map(lambda x: truncate_text(str(x), 28))
        st.markdown("##### Score final por alvo")
        st.bar_chart(chart, color="#8a6b2e")

    st.markdown("---")
    st.subheader("Ranking pré-calculado do projeto")
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

with tab_xai:
    st.markdown(
        "SHAP explica *por que* o RF atribui probabilidade de **alta atividade** "
        f"(MIC ≤ 3,4 µM). Treino atual: **{n_train}** pares MIC."
    )

    global_report = predictor.global_shap_report()
    baseline_path = ROOT / "data" / "processed" / "models" / "shap_global_baseline.json"
    baseline_report = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    )

    g1, g2 = st.columns(2)
    with g1:
        if global_report:
            st.markdown("**Global — multimodal**")
            st.caption(f"{global_report.get('n_samples')} MICs")
            st.pyplot(
                plot_global_importance(global_report["global_importance"]),
                clear_figure=True,
                use_container_width=True,
            )
    with g2:
        if baseline_report:
            st.markdown("**Global — baseline**")
            st.caption(f"{baseline_report.get('n_samples')} MICs")
            st.pyplot(
                plot_global_importance(
                    baseline_report["global_importance"],
                    title="Importância |SHAP| — baseline",
                ),
                clear_figure=True,
                use_container_width=True,
            )

    st.markdown("##### Beeswarm")
    st.caption("Cada ponto = um par MIC do treino. Cor = valor do descritor.")
    b1, b2 = st.columns(2)
    with b1:
        try:
            st.image(cached_beeswarm(True, n_train), caption="Multimodal", use_container_width=True)
        except Exception as e:
            st.error(f"Beeswarm multimodal: {e}")
    with b2:
        try:
            st.image(cached_beeswarm(False, n_train), caption="Baseline", use_container_width=True)
        except Exception as e:
            st.error(f"Beeswarm baseline: {e}")

    st.markdown("---")
    st.subheader("Explicação local")
    xai_seq = st.text_input("Sequência", value=st.session_state.get("seq_main", "FFSLIPSLVGGLISAFK"), key="xai_seq")
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

with tab_data:
    summary_path = ROOT / "data" / "processed" / "build_summary.json"
    bench_report = ROOT / "data" / "bench" / "import_report.json"
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Peptídeos (base)", f"{len(base):,}")
    d2.metric("Projeto", len(project_df) if not project_df.empty else "—")
    d3.metric("Pares", f"{len(pairs):,}")
    d4.metric("MICs", int(pairs["mic_value"].notna().sum()))

    if bench_report.exists():
        with st.expander("Relatório bancada"):
            st.json(json.loads(bench_report.read_text(encoding="utf-8")))
    if summary_path.exists():
        with st.expander("build_summary.json"):
            st.json(json.loads(summary_path.read_text(encoding="utf-8")))

    st.subheader("Peptídeos do projeto")
    st.caption("P01–P09 podem ter sequência placeholder (igual ao nativo) — preferir P10–P18 para testes.")
    if not project_df.empty:
        view = project_df[["peptide_id", "name", "sequence", "net_charge", "source"]].copy()
        # flag duplicates of native sequences among analogs
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

with tab_api:
    st.markdown(
        f"""
Endpoints locais (`uvicorn api.main:app --port 8001`):

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
  -d '{{"sequence":"FFSLIPKLVAGLISAFK","target_id":"S_aureus_ATCC29213","net_charge":3}}'
```
"""
    )
    if info:
        st.markdown("**Modelo carregado**")
        st.write(f"- Tipo: {info.get('model_type', '—')}")
        st.write(f"- Amostras: {info.get('n_samples', '—')}")
        st.write(f"- LOO AUC: {info.get('loo_auc', '—')}")
