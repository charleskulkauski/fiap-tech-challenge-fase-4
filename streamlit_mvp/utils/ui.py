from __future__ import annotations

import streamlit as st

from streamlit_mvp.utils.case_store import load_case_status


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def show_sidebar_navigation_guide() -> None:
    with st.sidebar:
        st.markdown("## Menu")
        st.page_link("Home.py", label="🏠 Home")
        st.page_link("pages/1_Entrada_do_Caso.py", label="🗂️ Entrada do Caso")
        st.page_link("pages/2_Processamento_e_Status.py", label="⚙️ Processamento e Status")
        st.page_link("pages/3_Resultados_Multimodais.py", label="📊 Resultados Multimodais")
        st.page_link("pages/4_Relatorio_e_Exportacao.py", label="📄 Relatório e Exportação")


def show_case_status_sidebar(case_id: str | None) -> None:
    if not case_id:
        st.sidebar.info("Nenhum caso selecionado.")
        return
    status = load_case_status(case_id)
    st.sidebar.markdown(f"### Caso ativo: `{case_id}`")
    st.sidebar.write(f"Criado: {'Sim' if status.get('case_created') else 'Não'}")
    st.sidebar.write(
        f"Processado: {'Sim' if status.get('processing_done') else 'Não'}"
    )
    st.sidebar.write(
        f"Relatório pronto: {'Sim' if status.get('report_ready') else 'Não'}"
    )
    st.sidebar.write(
        f"Encerrado: {'Sim' if status.get('case_closed') else 'Não'}"
    )
    if status.get("last_error"):
        st.sidebar.error(status["last_error"])
