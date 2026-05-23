from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

                                                                     
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_mvp.utils.case_policy import (
    CATEGORY_ACOLHIMENTO,
    CATEGORY_DOR,
    CATEGORY_LAPARO,
    get_case_policy,
    normalize_case_category,
)
from streamlit_mvp.utils.case_store import get_case_files, load_case_meta, load_case_status
from streamlit_mvp.utils.results_view import (
    build_top_events_table,
    highlight_keywords,
    load_summary_from_csv,
)
from streamlit_mvp.utils.ui import (
    page_header,
    show_case_status_sidebar,
    show_sidebar_navigation_guide,
)

st.set_page_config(page_title="Resultados Multimodais", page_icon="📊", layout="wide")

page_header(
    "📊 Resultados Multimodais",
    "Inspeção rápida das evidências de vídeo, áudio e eventos críticos.",
)
show_sidebar_navigation_guide()

case_id = st.session_state.get("case_id")
show_case_status_sidebar(case_id)

if not case_id:
    st.warning("Página bloqueada: crie um caso em Entrada do Caso.")
    st.stop()

status = load_case_status(case_id)
if not status.get("processing_done"):
    st.warning("Página bloqueada: execute o pipeline em Processamento e Status.")
    st.stop()

files = get_case_files(case_id)
case_meta = load_case_meta(case_id)
category = normalize_case_category(case_meta.get("categoria_caso"))
policy = get_case_policy(category)

category_badge = {
    CATEGORY_ACOLHIMENTO: "🟣 Acolhimento/Violência",
    CATEGORY_DOR: "🟠 Dor corporal",
    CATEGORY_LAPARO: "🔵 Laparoscopia ginecológica",
}.get(category, "⚪ Indeterminado")
st.markdown(f"### Categoria ativa: {category_badge}")

if category == CATEGORY_ACOLHIMENTO:
    st.info(
        "Modo ativo: foco em sinais emocionais/comportamentais, sangue facial e postura."
    )
elif category == CATEGORY_DOR:
    st.info(
        "Modo ativo: foco em dor corporal, expressão de sofrimento, postura e análise de áudio."
    )
elif category == CATEGORY_LAPARO:
    st.info(
        "Modo ativo: foco laparoscópico interno (instrumentos/sangramento/contrações)."
    )

events_df = pd.DataFrame()
if files.video_events_csv.is_file():
    events_df = build_top_events_table(
        files.video_events_csv,
        instrument_csv=files.video_instrument_csv,
        top_n=20,
        include_hematoma=policy.allow_bruise_detection,
    )

st.subheader("Bloco Vídeo")
if not files.video_out.is_file():
    st.info("Sem saída de vídeo para este caso (modo áudio-only).")
elif files.video_out.stat().st_size <= 0:
    st.warning("`video_out.mp4` foi gerado, mas está vazio.")
else:
    _, video_col, _ = st.columns([1, 2, 1])
    with video_col:
        st.video(files.video_out.read_bytes(), format="video/mp4")

if files.video_events_csv.is_file():
    summary = load_summary_from_csv(
        files.video_events_csv,
        files.video_instrument_csv,
        include_hematoma=policy.allow_bruise_detection,
        category=category,
    )
    if policy.allow_pain_and_pose and not policy.allow_instrument_identification:
        if policy.allow_bruise_detection:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Dor máxima", f"{summary['pain_smoothed_max']:.1f}%")
            c2.metric("Nº de picos", str(summary["pain_peaks_count"]))
            c3.metric("Flags posturais", str(len(summary["flags"])))
            c4.metric("Sangue identificado", str(summary["sangue_identificado"]))
            c5.metric("Hematoma identificado", str(summary["hematoma_identificado"]))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dor máxima", f"{summary['pain_smoothed_max']:.1f}%")
            c2.metric("Nº de picos", str(summary["pain_peaks_count"]))
            c3.metric("Flags posturais", str(len(summary["flags"])))
            c4.metric("Sangue identificado", str(summary["sangue_identificado"]))
    elif policy.allow_instrument_identification:
        qtd_tabela = 0
        sangue_identificado = "Não"
        if not events_df.empty and "qtd_instrumentos" in events_df.columns:
            qtd_tabela = int(
                pd.to_numeric(events_df["qtd_instrumentos"], errors="coerce")
                .fillna(0.0)
                .max()
            )
        if not events_df.empty and "flag_sangue_identificado" in events_df.columns:
            sangue_identificado = (
                "Sim"
                if events_df["flag_sangue_identificado"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"sim", "1", "true"})
                .any()
                else "Não"
            )
        c1, c2 = st.columns(2)
        c1.metric("Quantidade de instrumentos", str(qtd_tabela))
        c2.metric("Presença de Sangue identificado", sangue_identificado)
else:
    st.warning("`video_events.csv` não encontrado. Métricas de vídeo indisponíveis.")

if policy.allow_audio_transcript:
    st.subheader("Transcrição - Azure Speech")
    if files.transcript_txt.is_file():
        transcript = files.transcript_txt.read_text(encoding="utf-8")
        if transcript.strip():
            st.markdown(
                highlight_keywords(transcript),
                unsafe_allow_html=True,
            )
        else:
            st.info("Transcrição vazia.")
    else:
        st.warning("Transcrição não encontrada.")

    if files.audio_analysis_json.is_file():
        with st.expander("Análise de áudio (JSON técnico)"):
            try:
                audio_payload = json.loads(files.audio_analysis_json.read_text(encoding="utf-8"))
                st.json(audio_payload)
            except json.JSONDecodeError:
                st.warning("Não foi possível interpretar o JSON de análise de áudio.")
    else:
        st.warning("Análise de áudio não encontrada.")
else:
    st.info("Bloco de áudio oculto para a categoria ativa.")

st.subheader("Eventos Identificados")
if not files.video_events_csv.is_file():
    st.info("Eventos de vídeo indisponíveis para este caso.")
else:
    if events_df.empty:
        st.info("Sem eventos para exibir.")
    else:
        if category in {CATEGORY_ACOLHIMENTO, CATEGORY_DOR}:
            drop_cols = [
                "qtd_instrumentos",
                "instrumentos_detectados",
            ]
            events_df = events_df.drop(
                columns=[col for col in drop_cols if col in events_df.columns],
                errors="ignore",
            )
        elif category == CATEGORY_LAPARO:
            keep_cols = [
                "tempo_video_s",
                "qtd_instrumentos",
                "instrumentos_detectados",
                "score_sangue_identificado",
                "flag_sangue_identificado",
                "faixa_prioridade_evento",
            ]
            events_df = events_df[[col for col in keep_cols if col in events_df.columns]]

        st.dataframe(events_df, use_container_width=True, hide_index=True)
        if category in {CATEGORY_ACOLHIMENTO, CATEGORY_DOR}:
            st.caption(
                "Faixas operacionais: dor baixa <25, moderada 25-39.9, alta >=40; "
                "postural baixa <30, moderada 30-59.9, alta >=60; "
                "prioridade do evento baixa <25, moderada 25-44.9, alta >=45."
            )
        else:
            st.caption(
                "Tabela filtrada para contexto laparoscópico (instrumentos/sangramento/prioridade)."
            )

st.subheader("Prontuario")
if files.risk_alert_json.is_file():
    st.markdown("**Análise do Video (JSON Técnico)**")
    try:
        risk_payload = json.loads(files.risk_alert_json.read_text(encoding="utf-8"))
        st.json(risk_payload)
        vision_payload = risk_payload.get("vision", {}) if isinstance(risk_payload, dict) else {}
        emotion_agg = (
            vision_payload.get("emotion_probabilities", {})
            if isinstance(vision_payload, dict)
            else {}
        )
        if category == CATEGORY_DOR and isinstance(emotion_agg, dict) and emotion_agg:
            st.markdown("**Agregação Emocional Temporal**")
            emotion_rows: list[dict[str, float | str]] = []
            for emotion in ("angry", "disgust", "fear", "happy", "neutral", "sad"):
                values = emotion_agg.get(emotion, {})
                if not isinstance(values, dict):
                    values = {}
                emotion_rows.append(
                    {
                        "emoção": emotion,
                        "média": float(values.get("mean", 0.0) or 0.0),
                        "máximo": float(values.get("max", 0.0) or 0.0),
                        "último_frame": float(values.get("last", 0.0) or 0.0),
                    }
                )
            emotion_df = pd.DataFrame(emotion_rows)
            st.dataframe(emotion_df, use_container_width=True, hide_index=True)
    except json.JSONDecodeError:
        st.warning("Não foi possível interpretar o `risk_alert.json`.")
else:
    st.warning("`risk_alert.json` não encontrado.")

if files.prontuario_json.is_file():
    with st.expander("Análise de audio + vídeo (JSON técnico)"):
        try:
            prontuario_payload = json.loads(files.prontuario_json.read_text(encoding="utf-8"))
            st.json(prontuario_payload)
        except json.JSONDecodeError:
            st.warning("Não foi possível interpretar o prontuário multimodal.")
else:
    st.warning("Prontuário multimodal em JSON não encontrado.")

if st.button("➡️ Ir para Relatório e Exportação"):
    st.switch_page("pages/4_Relatorio_e_Exportacao.py")
