"""PepMem-AI Streamlit dashboard (PoC)."""

from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

from pepmem.paths import project_root

ROOT = project_root()
sys.path.insert(0, str(ROOT))

from pepmem.predictor import PepMemPredictor
from pepmem.shap_explain import plot_beeswarm, plot_contributions, plot_global_importance

st.set_page_config(
    page_title="PepMem-AI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      html, body {
        overflow-x: clip !important;
        max-width: 100%;
        margin: 0;
        padding: 0;
      }
      .stApp, [data-testid="stAppViewContainer"], .main, section.main {
        overflow-x: clip !important;
        max-width: 100% !important;
        width: 100% !important;
      }
      [data-testid="stAppViewContainer"] > section.main > div.block-container {
        max-width: 720px !important;
        width: 100% !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
      }
      [data-testid="stHorizontalBlock"],
      [data-testid="stVerticalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        gap: 0.5rem;
      }
      [data-testid="column"] {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden;
      }
      [data-testid="stImage"] img,
      [data-testid="stPyplot"] img,
      [data-testid="stPyplot"] svg,
      [data-testid="stVegaLiteChart"] {
        max-width: 100% !important;
        width: 100% !important;
        height: auto !important;
      }
      [data-testid="stVegaLiteChart"] > div {
        max-width: 100% !important;
        overflow: hidden !important;
      }
      [data-testid="stTable"] {
        max-width: 100%;
        overflow-x: hidden;
      }
      [data-testid="stTable"] table {
        width: 100%;
        table-layout: fixed;
      }
      [data-testid="stTable"] td,
      [data-testid="stTable"] th {
        word-break: break-word;
        overflow-wrap: anywhere;
        white-space: normal;
        font-size: 0.85rem;
      }
      [data-testid="stMetric"] {
        min-width: 0;
        overflow: hidden;
      }
      [data-testid="stMetricLabel"] {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        gap: 0.25rem;
      }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div,
      [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      [data-testid="stCodeBlock"] pre {
        white-space: pre-wrap;
        word-break: break-all;
      }
      .stMarkdown table {
        width: 100%;
        table-layout: fixed;
      }
      .stMarkdown table td,
      .stMarkdown table th {
        word-break: break-word;
        overflow-wrap: anywhere;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PepMem-AI")
st.caption("PoC — predição de interação peptídeo–membrana (InovAI Lab / UFRN)")


@st.cache_resource
def get_predictor() -> PepMemPredictor:
    return PepMemPredictor(use_embeddings=True)


@st.cache_data(show_spinner="Gerando beeswarm SHAP...")
def cached_beeswarm(use_embeddings: bool, _layout_version: int = 2) -> bytes:
    import io

    import joblib
    import matplotlib.pyplot as plt

    fname = "multimodal_mic_rf.joblib" if use_embeddings else "baseline_mic_rf.joblib"
    path = ROOT / "data" / "processed" / "models" / fname
    if not path.exists():
        raise FileNotFoundError(f"Modelo ausente: {fname}")
    pipe = joblib.load(path)
    title = (
        "Beeswarm SHAP — multimodal (12 MICs)"
        if use_embeddings
        else "Beeswarm SHAP — baseline (12 MICs)"
    )
    fig = plot_beeswarm(pipe, use_embeddings, title=title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="Calculando SHAP...")
def cached_explain(sequence: str, target_id: str, net_charge: float | None) -> dict:
    return get_predictor().explain_pair(sequence, target_id, net_charge=net_charge)


predictor = get_predictor()
targets = predictor.targets
target_options = targets.set_index("target_id")["target"].to_dict()


def format_target_label(target_id: str) -> str:
    name = target_options.get(target_id, target_id)
    short = name if len(name) <= 32 else f"{name[:29]}…"
    return f"{short} ({target_id})"


def truncate_text(value: object, max_len: int = 40) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def show_table(df: pd.DataFrame, max_text_len: int = 40) -> None:
    """Tabela HTML estática — evita redimensionamento lateral do st.dataframe."""
    view = df.copy()
    for col in view.columns:
        if view[col].dtype == object:
            view[col] = view[col].map(lambda x: truncate_text(x, max_text_len))
    st.table(view)


def show_metrics_quad(items: list[tuple[str, str]]) -> None:
    """Métricas 2×2 — evita overflow de 4 colunas em 100% zoom."""
    row_a, row_b = st.columns(2), st.columns(2)
    for col, (label, value) in zip([*row_a, *row_b], items):
        col.metric(label, value)


def render_shap_block(expl: dict, target_label: str) -> None:
    st.metric("Prob. alta atividade (modelo RF)", f"{expl['pred_high_activity_prob']:.1%}")
    fig = plot_contributions(
        expl["shap_contributions"],
        title=f"Contribuições SHAP — {target_label}",
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.markdown("**Tabela de contribuições**")
    df = pd.DataFrame(expl["shap_contributions"])[["label", "shap_value", "group"]]
    df["shap_value"] = df["shap_value"].map(lambda x: f"{x:+.4f}")
    show_table(df, max_text_len=36)
    st.caption("Verde = favorece alta atividade · Vermelho = desfavorece")


tab_pred, tab_rank, tab_xai, tab_data, tab_api = st.tabs(
    ["Predição", "Ranking", "XAI (SHAP)", "Datasets", "API"]
)

with tab_pred:
    sequence = st.text_input(
        "Sequência peptídica",
        value="FFSLIPKLVKGLISAFK",
        help="Aminoácidos em letra única (ex.: StigA6)",
    )
    net_charge = st.number_input("Carga líquida (opcional)", value=3.0, step=1.0, format="%.1f")
    use_charge = st.checkbox("Informar carga manualmente", value=True)
    charge = float(net_charge) if use_charge else None

    target_id = st.selectbox(
        "Membrana-alvo",
        options=list(target_options.keys()),
        format_func=format_target_label,
    )

    if st.button("Predizer", type="primary"):
        with st.spinner("Calculando descritores, PMI e modelo..."):
            try:
                res = predictor.predict_pair(sequence, target_id, net_charge=charge)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        show_metrics_quad([
            ("PMI", f"{res['pmi']:.3f}"),
            ("Prob. alta atividade", f"{res['pred_high_activity_prob']:.1%}"),
            ("Carga peptídeo", f"{res['q_peptide']:.1f}"),
            ("Hidrofobicidade", f"{res['h_peptide']:.3f}"),
        ])

        st.subheader("Detalhes")
        with st.expander("Ver resposta completa (JSON)", expanded=False):
            st.json({k: v for k, v in res.items() if k != "sequence"})
        st.code(res["sequence"], language=None)

        st.subheader("Explicação SHAP")
        try:
            expl = cached_explain(sequence, target_id, charge)
            render_shap_block(expl, target_options[target_id])
        except Exception as e:
            st.error(f"SHAP indisponível: {e}")

with tab_rank:
    seq_rank = st.text_input("Sequência para ranking", value="FFSLIPKLVKGLISAFK", key="rank_seq")
    selected = st.multiselect(
        "Alvos (vazio = todos)",
        options=list(target_options.keys()),
        format_func=lambda x: truncate_text(target_options[x], 36),
    )
    lam = st.slider("Penalização toxicidade (λ)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Gerar ranking"):
        df = predictor.rank_peptide(seq_rank, target_ids=selected or None, lambda_tox=lam)
        show = df[
            [
                "target_id", "target_type", "pmi", "pmi_sel",
                "pred_high_activity_prob", "final_score",
            ]
        ].copy()
        show["pred_high_activity_prob"] = show["pred_high_activity_prob"].map(lambda x: f"{x:.1%}")
        show_table(show, max_text_len=28)

        chart = df.set_index("target_id")["final_score"].dropna().copy()
        chart.index = chart.index.map(lambda x: truncate_text(str(x), 24))
        st.bar_chart(chart)

    st.subheader("Peptídeos do projeto (pré-calculado)")
    ranking_path = ROOT / "data" / "processed" / "models" / "project_ranking_baseline.csv"
    if ranking_path.exists():
        pre = pd.read_csv(ranking_path)
        filt = st.selectbox("Filtrar alvo", ["Todos"] + sorted(pre["target_id"].unique().tolist()))
        view = pre if filt == "Todos" else pre[pre["target_id"] == filt]
        show_table(
            view.nlargest(20, "pred_high_activity_prob")[
                ["peptide_id", "target_id", "pmi", "pmi_sel", "pred_high_activity_prob"]
            ],
            max_text_len=28,
        )

with tab_xai:
    st.markdown(
        """
        **SHAP** explica a predição do Random Forest: quanto cada descritor empurra a probabilidade
        de **alta atividade** (MIC ≤ 3,4 µM). Azul = descritores clássicos · Roxo = ESM-2 agregado.
        """
    )

    global_report = predictor.global_shap_report()
    baseline_report = None
    baseline_path = ROOT / "data" / "processed" / "models" / "shap_global_baseline.json"
    if baseline_path.exists():
        import json

        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))

    if global_report or baseline_report:
        if global_report:
            st.subheader("Importância global — multimodal (ESM-2 + clássicas)")
            st.caption(
                f"{global_report.get('model')} · {global_report.get('n_samples')} MICs de treino"
            )
            gfig = plot_global_importance(global_report["global_importance"])
            st.pyplot(gfig, clear_figure=True, use_container_width=True)
        if baseline_report:
            st.subheader("Importância global — baseline (só clássicas)")
            st.caption(
                f"{baseline_report.get('model')} · {baseline_report.get('n_samples')} MICs de treino"
            )
            bfig = plot_global_importance(
                baseline_report["global_importance"],
                title="Importância global |SHAP| — baseline",
            )
            st.pyplot(bfig, clear_figure=True, use_container_width=True)
    else:
        st.warning("Execute `python scripts/compute_shap.py` para gerar os gráficos globais.")

    st.subheader("Beeswarm SHAP (visão clássica)")
    st.caption(
        "Cada ponto é um par peptídeo–membrana do treino (12 MICs). "
        "Cor = valor do descritor · posição horizontal = impacto SHAP na predição."
    )
    try:
        st.image(
            cached_beeswarm(True, _layout_version=4),
            caption="Multimodal (clássicas + ESM-2 agregado)",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Beeswarm multimodal: {e}")
    try:
        st.image(
            cached_beeswarm(False, _layout_version=4),
            caption="Baseline (só descritores clássicos)",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Beeswarm baseline: {e}")

    st.divider()
    st.subheader("Explicação local — seu peptídeo")
    xai_seq = st.text_input("Sequência", value="FFSLIPSLVGGLISAFK", key="xai_seq")
    xai_charge = st.number_input("Carga líquida", value=3.0, step=1.0, key="xai_charge")
    xai_use_charge = st.checkbox("Informar carga", value=True, key="xai_use_charge")
    xai_target = st.selectbox(
        "Membrana-alvo",
        options=list(target_options.keys()),
        format_func=format_target_label,
        key="xai_target",
        index=list(target_options.keys()).index("E_coli_ATCC25922")
        if "E_coli_ATCC25922" in target_options
        else 0,
    )

    charge_xai = float(xai_charge) if xai_use_charge else None
    explain_now = st.button("Recalcular SHAP", type="primary", key="xai_btn")

    if explain_now:
        try:
            expl = cached_explain(xai_seq, xai_target, charge_xai)
            render_shap_block(expl, target_options[xai_target])
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Clique em **Recalcular SHAP** para gerar a explicação local.")

with tab_data:
    summary_path = ROOT / "data" / "processed" / "build_summary.json"
    bench_report = ROOT / "data" / "bench" / "import_report.json"
    if summary_path.exists():
        with st.expander("Resumo do build (JSON)", expanded=False):
            st.json(summary_path.read_text(encoding="utf-8"))
    if bench_report.exists():
        st.subheader("Bancada (MICs importados)")
        with st.expander("Relatório de importação", expanded=False):
            st.json(bench_report.read_text(encoding="utf-8"))
        st.caption("Edite `data/bench/mic_bench.csv` e rode `python scripts/import_bench_mic.py --retrain`")
    else:
        st.info(
            "Para inserir MICs da bancada: preencha `data/bench/mic_bench.csv` "
            "e execute `python scripts/import_bench_mic.py --retrain`"
        )
    base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    dm1, dm2 = st.columns(2)
    dm3, _ = st.columns(2)
    dm1.metric("Peptídeos (base)", len(base))
    dm2.metric("Pares peptídeo–membrana", len(pairs))
    dm3.metric("MICs experimentais", int(pairs["mic_value"].notna().sum()))
    st.subheader("Peptídeos do projeto")
    proj = pd.read_csv(ROOT / "data" / "processed" / "pepmem_base_project.csv")
    show_table(proj[["peptide_id", "name", "sequence", "net_charge"]], max_text_len=36)

with tab_api:
    st.markdown(
        """
        ### Endpoints locais

        ```bash
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
        ```

        | Método | Rota | Descrição |
        |--------|------|-----------|
        | GET | `/health` | Status |
        | GET | `/targets` | Lista de membranas-alvo |
        | POST | `/predict` | Predição para um par |
        | POST | `/explain` | Explicação SHAP local |
        | GET | `/explain/global` | Importância SHAP global |
        | POST | `/rank` | Ranking multi-alvo |

        **Exemplo:**
        ```bash
        curl -X POST http://localhost:8001/rank \\
          -H "Content-Type: application/json" \\
          -d '{"sequence":"FFSLIPKLVKGLISAFK"}'
        ```
        """
    )
    info = predictor.model_info
    if info:
        st.subheader("Modelo")
        st.write(f"Tipo: {info.get('model_type', 'baseline RF')}")
        st.write(f"Amostras treino: {info.get('n_samples', '—')}")
        st.write(f"LOO AUC: {info.get('loo_auc', '—')}")
