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

st.set_page_config(page_title="PepMem-AI", page_icon="🧬", layout="wide")

st.title("PepMem-AI")
st.caption("PoC — predição de interação peptídeo–membrana (InovAI Lab / UFRN)")


@st.cache_resource
def get_predictor() -> PepMemPredictor:
    return PepMemPredictor(use_embeddings=True)


@st.cache_data(show_spinner="Gerando beeswarm SHAP...")
def cached_beeswarm(use_embeddings: bool) -> bytes:
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
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="Calculando SHAP...")
def cached_explain(sequence: str, target_id: str, net_charge: float | None) -> dict:
    return get_predictor().explain_pair(sequence, target_id, net_charge=net_charge)


def render_shap_block(expl: dict, target_label: str) -> None:
    st.metric("Prob. alta atividade (modelo RF)", f"{expl['pred_high_activity_prob']:.1%}")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = plot_contributions(
            expl["shap_contributions"],
            title=f"Contribuições SHAP — {target_label}",
        )
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    with c2:
        st.markdown("**Tabela de contribuições**")
        df = pd.DataFrame(expl["shap_contributions"])[["label", "shap_value", "group"]]
        df["shap_value"] = df["shap_value"].map(lambda x: f"{x:+.4f}")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("Verde = favorece alta atividade · Vermelho = desfavorece")


predictor = get_predictor()
targets = predictor.targets
target_options = targets.set_index("target_id")["target"].to_dict()

tab_pred, tab_rank, tab_xai, tab_data, tab_api = st.tabs(
    ["Predição", "Ranking", "XAI (SHAP)", "Datasets", "API"]
)

with tab_pred:
    col1, col2 = st.columns([2, 1])
    with col1:
        sequence = st.text_input(
            "Sequência peptídica",
            value="FFSLIPKLVKGLISAFK",
            help="Aminoácidos em letra única (ex.: StigA6)",
        )
    with col2:
        net_charge = st.number_input("Carga líquida (opcional)", value=3.0, step=1.0, format="%.1f")
        use_charge = st.checkbox("Informar carga manualmente", value=True)
        charge = float(net_charge) if use_charge else None

    target_id = st.selectbox(
        "Membrana-alvo",
        options=list(target_options.keys()),
        format_func=lambda x: f"{target_options[x]} ({x})",
    )

    if st.button("Predizer", type="primary"):
        with st.spinner("Calculando descritores, PMI e modelo..."):
            try:
                res = predictor.predict_pair(sequence, target_id, net_charge=charge)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PMI", f"{res['pmi']:.3f}")
        c2.metric("Prob. alta atividade", f"{res['pred_high_activity_prob']:.1%}")
        c3.metric("Carga peptídeo", f"{res['q_peptide']:.1f}")
        c4.metric("Hidrofobicidade", f"{res['h_peptide']:.3f}")

        st.subheader("Detalhes")
        st.json({k: v for k, v in res.items() if k != "sequence"})
        st.code(res["sequence"])

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
        format_func=lambda x: target_options[x],
    )
    lam = st.slider("Penalização toxicidade (λ)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Gerar ranking"):
        df = predictor.rank_peptide(seq_rank, target_ids=selected or None, lambda_tox=lam)
        show = df[
            [
                "target", "target_type", "pmi", "pmi_sel",
                "pred_high_activity_prob", "final_score",
            ]
        ].copy()
        show["pred_high_activity_prob"] = show["pred_high_activity_prob"].map(lambda x: f"{x:.1%}")
        st.dataframe(show, width="stretch", hide_index=True)

        st.bar_chart(df.set_index("target")["final_score"].dropna())

    st.subheader("Peptídeos do projeto (pré-calculado)")
    ranking_path = ROOT / "data" / "processed" / "models" / "project_ranking_baseline.csv"
    if ranking_path.exists():
        pre = pd.read_csv(ranking_path)
        filt = st.selectbox("Filtrar alvo", ["Todos"] + sorted(pre["target_id"].unique().tolist()))
        view = pre if filt == "Todos" else pre[pre["target_id"] == filt]
        st.dataframe(
            view.nlargest(20, "pred_high_activity_prob")[
                ["peptide_id", "target", "pmi", "pmi_sel", "pred_high_activity_prob"]
            ],
            width="stretch",
            hide_index=True,
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
        g1, g2 = st.columns(2)
        if global_report:
            with g1:
                st.subheader("Importância global — multimodal (ESM-2 + clássicas)")
                st.caption(
                    f"{global_report.get('model')} · {global_report.get('n_samples')} MICs de treino"
                )
                gfig = plot_global_importance(global_report["global_importance"])
                st.pyplot(gfig, clear_figure=True, use_container_width=True)
        if baseline_report:
            with g2:
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
    b1, b2 = st.columns(2)
    try:
        with b1:
            st.image(cached_beeswarm(True), caption="Multimodal (clássicas + ESM-2 agregado)", width="stretch")
    except Exception as e:
        st.error(f"Beeswarm multimodal: {e}")
    try:
        with b2:
            st.image(cached_beeswarm(False), caption="Baseline (só descritores clássicos)", width="stretch")
    except Exception as e:
        st.error(f"Beeswarm baseline: {e}")

    st.divider()
    st.subheader("Explicação local — seu peptídeo")
    xai_col1, xai_col2 = st.columns([2, 1])
    with xai_col1:
        xai_seq = st.text_input("Sequência", value="FFSLIPSLVGGLISAFK", key="xai_seq")
    with xai_col2:
        xai_charge = st.number_input("Carga líquida", value=3.0, step=1.0, key="xai_charge")
        xai_use_charge = st.checkbox("Informar carga", value=True, key="xai_use_charge")
    xai_target = st.selectbox(
        "Membrana-alvo",
        options=list(target_options.keys()),
        format_func=lambda x: f"{target_options[x]} ({x})",
        key="xai_target",
        index=list(target_options.keys()).index("E_coli_ATCC25922")
        if "E_coli_ATCC25922" in target_options
        else 0,
    )

    charge_xai = float(xai_charge) if xai_use_charge else None
    explain_now = st.button("Recalcular SHAP", type="primary", key="xai_btn")

    if explain_now or xai_seq:
        try:
            expl = cached_explain(xai_seq, xai_target, charge_xai)
            render_shap_block(expl, target_options[xai_target])
        except Exception as e:
            st.error(str(e))

with tab_data:
    summary_path = ROOT / "data" / "processed" / "build_summary.json"
    if summary_path.exists():
        st.json(summary_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    base = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_base.parquet")
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pepmem_pairs.parquet")
    c1.metric("Peptídeos (base)", len(base))
    c2.metric("Pares peptídeo–membrana", len(pairs))
    c3.metric("MICs experimentais", int(pairs["mic_value"].notna().sum()))
    st.subheader("Peptídeos do projeto")
    proj = pd.read_csv(ROOT / "data" / "processed" / "pepmem_base_project.csv")
    st.dataframe(proj[["peptide_id", "name", "sequence", "net_charge"]], hide_index=True)

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
