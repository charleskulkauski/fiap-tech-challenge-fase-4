from __future__ import annotations

import traceback
import sys
from pathlib import Path

import streamlit as st

                                                                     
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_mvp.utils.case_store import (
    get_case_files,
    load_case_status,
    save_case_status,
)
from streamlit_mvp.utils.pipeline_runner import run_case_pipeline
from models.lgpd_compliance import LgpdProcessingBlockedError
from streamlit_mvp.utils.ui import (
    page_header,
    show_case_status_sidebar,
    show_sidebar_navigation_guide,
)

st.set_page_config(page_title="Processamento e Status", page_icon="⚙️", layout="wide")

page_header(
    "⚙️ Processamento e Status",
    "Execução clínica/operacional do pipeline multimodal com feedback de progresso.",
)
show_sidebar_navigation_guide()

case_id = st.session_state.get("case_id")
show_case_status_sidebar(case_id)

if not case_id:
    st.warning("Página bloqueada: crie um caso em Entrada do Caso.")
    st.stop()

status = load_case_status(case_id)
if not status.get("case_created"):
    st.warning("Página bloqueada: o caso ainda não foi criado corretamente.")
    st.stop()

files = get_case_files(case_id)
st.write(f"Caso ativo: `{case_id}`")

save_json = st.checkbox("Salvar resumo estruturado em JSON (opcional)", value=True)

col_a, col_b = st.columns(2)
run_clicked = col_a.button("Executar processamento", type="primary")
reprocess_clicked = col_b.button("Reprocessar")

if run_clicked or reprocess_clicked:
    has_video = files.video_input.is_file()
    has_audio = files.audio_input.is_file()
    progress = st.progress(0, text="Preparando execução...")
    stage_box = st.empty()

    def on_stage(stage_name: str) -> None:
        if stage_name == "video":
            progress.progress(20, text="Executando análise de vídeo...")
        elif stage_name == "audio":
            progress.progress(40, text="Executando análise de áudio...")
        elif stage_name == "risk":
            progress.progress(60, text="Consolidando risco visual + psicológico...")
        elif stage_name == "prontuario":
            progress.progress(80, text="Gerando prontuário auditável...")
        elif stage_name == "report":
            progress.progress(95, text="Gerando relatório clínico...")
        elif stage_name == "done":
            progress.progress(100, text="Pipeline concluído.")

    try:
        run_case_pipeline(
            case_id=case_id,
            save_json=save_json,
            stage_callback=on_stage,
        )

        stage_box.info("Artefatos gerados:")
        if has_video:
            st.markdown(f"- `{files.video_out}`")
            st.markdown(f"- `{files.video_events_csv}`")
        if has_audio:
            st.markdown(f"- `{files.transcript_txt}`")
            st.markdown(f"- `{files.audio_analysis_json}`")
    except LgpdProcessingBlockedError as exc:
        save_case_status(
            case_id,
            {
                "processing_done": False,
                "report_ready": False,
                "last_error": str(exc),
                "step": "failed",
            },
        )
        progress.progress(100, text="Processamento bloqueado por LGPD.")
        stage_box.error(
            f"Processamento bloqueado por política LGPD: {exc} "
            "Revise o consentimento em Entrada do Caso."
        )
    except Exception as exc:
        save_case_status(
            case_id,
            {
                "processing_done": False,
                "report_ready": False,
                "last_error": str(exc),
                "step": "failed",
            },
        )
        progress.progress(100, text="Falha na execução.")
        stage_box.error(f"Erro durante o processamento: {exc}")
        with st.expander("Detalhes técnicos"):
            st.code(traceback.format_exc())

status = load_case_status(case_id)
if status.get("processing_done"):
    st.success("Status final: processamento concluído.")
    if st.button("Ir para Resultados Multimodais ➡️"):
        st.switch_page("pages/3_Resultados_Multimodais.py")
else:
    st.info("Aguardando processamento para liberar Resultados Multimodais.")
