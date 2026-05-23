from __future__ import annotations

import base64
from datetime import datetime
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

                                                                     
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_mvp.utils.case_store import (
    get_case_files,
    load_case_meta,
    load_case_status,
    save_case_meta,
    save_case_status,
)
from streamlit_mvp.utils.report_export import (
    build_final_report_markdown,
    export_pdf_from_text,
)
from streamlit_mvp.utils.ui import (
    page_header,
    show_case_status_sidebar,
    show_sidebar_navigation_guide,
)

st.set_page_config(page_title="Relatório e Exportação", page_icon="📄", layout="wide")

page_header(
    "📄 Relatório e Exportação",
    "Revisão final, complemento médico, exportação e encerramento do caso.",
)
show_sidebar_navigation_guide()

case_id = st.session_state.get("case_id")
show_case_status_sidebar(case_id)

if not case_id:
    st.warning("Página bloqueada: crie um caso em Entrada do Caso.")
    st.stop()

status = load_case_status(case_id)
if not status.get("processing_done"):
    st.warning("Página bloqueada: finalize o processamento em Processamento e Status.")
    st.stop()

files = get_case_files(case_id)
report_candidates = [
    files.report_pdf,
]
report_candidates.extend(
    sorted(files.reports_dir.glob("relatorio_*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)
)
report_source: Path | None = next((p for p in report_candidates if p.is_file()), None)
if report_source is None:
    st.warning("Relatório não encontrado para este caso.")
    st.info("Nenhum arquivo `relatorio.pdf` / `relatorio_*.pdf` encontrado em reports.")

    can_generate_now = files.video_events_csv.is_file() or files.transcript_txt.is_file()
    if not can_generate_now:
        st.error("Também não há `video_events.csv` nem transcrição para gerar o relatório agora.")
        st.stop()

    if st.button("🧾 Gerar relatório agora"):
        try:
            from models.report_generator import generate_report

            generated_path = generate_report(
                transcript_path=files.transcript_txt if files.transcript_txt.is_file() else None,
                events_csv_path=files.video_events_csv if files.video_events_csv.is_file() else None,
                instruments_csv_path=(
                    files.video_instrument_csv if files.video_instrument_csv.is_file() else None
                ),
                output_dir=files.reports_dir,
                consultation_start=datetime.now(),
                output_path=files.report_pdf,
            )
            st.success(f"Relatório PDF gerado com sucesso em `{generated_path}`.")
            st.rerun()
        except Exception as exc:
            st.error(f"Falha ao gerar relatório automaticamente: {exc}")
    st.stop()

from models.report_generator import generate_report_markdown_content

base_report = generate_report_markdown_content(
    transcript_path=files.transcript_txt if files.transcript_txt.is_file() else None,
    events_csv_path=files.video_events_csv if files.video_events_csv.is_file() else None,
    instruments_csv_path=files.video_instrument_csv if files.video_instrument_csv.is_file() else None,
)

st.caption(f"PDF base disponível: `{report_source.name}`")
st.markdown(base_report)

st.subheader("Complemento do médico")
complemento = st.text_area(
    "Texto livre",
    placeholder="Inclua observações clínicas complementares para o relatório final.",
    height=180,
)

c1, c2 = st.columns(2)
with c1:
    chk_nao_diag = st.checkbox("Confirmo que o conteúdo é apoio não diagnóstico.")
with c2:
    chk_revisado = st.checkbox("Confirmo que os dados foram revisados.")

if st.button("Exportar PDF", type="primary"):
    if not (chk_nao_diag and chk_revisado):
        st.error("Marque os dois checklists antes de exportar.")
    else:
        acknowledged_at = datetime.now()
        acknowledgment = {
            "acknowledged_non_diagnostic": True,
            "acknowledged_reviewed": True,
            "acknowledged_at": acknowledged_at.isoformat(timespec="seconds"),
            "case_id": case_id,
        }
        final_md = build_final_report_markdown(
            base_report,
            complemento,
            acknowledgment=acknowledgment,
        )
        final_pdf_path = files.report_pdf

        try:
            export_pdf_from_text(final_md, final_pdf_path)
            case_meta = load_case_meta(case_id)
            case_meta["export_acknowledgment"] = acknowledgment
            save_case_meta(case_id, case_meta)
            st.success(f"PDF exportado em `{final_pdf_path}`")
            if final_pdf_path.is_file():
                pdf_bytes = final_pdf_path.read_bytes()
                pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
                safe_case_id = "".join(
                    ch if ch.isalnum() or ch in {"-", "_"} else "_"
                    for ch in str(case_id)
                )
                download_name = f"relatorio_final_{safe_case_id}.pdf"
                components.html(
                    f"""
                    <a id="auto-download-link"
                       href="data:application/pdf;base64,{pdf_b64}"
                       download="{download_name}"></a>
                    <script>
                      document.getElementById("auto-download-link").click();
                    </script>
                    """,
                    height=0,
                )
        except RuntimeError as exc:
            st.warning(str(exc))

if st.button("Encerrar caso"):
    save_case_status(
        case_id,
        {
            "case_closed": True,
            "closed_at": datetime.now().isoformat(timespec="seconds"),
            "last_error": "",
        },
    )
    st.success("Caso marcado como finalizado.")
