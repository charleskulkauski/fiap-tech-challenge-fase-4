from __future__ import annotations
from pathlib import Path

import streamlit as st
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_mvp.utils.ui import (
    page_header,
    show_case_status_sidebar,
    show_sidebar_navigation_guide,
)

try:
    from src.hf_assets import ensure_models

    @st.cache_resource(show_spinner="Verificando modelos (Hugging Face)...")
    def _bootstrap_models() -> bool:
        return ensure_models()

    if not _bootstrap_models():
        st.warning(
            "Modelos YOLO não encontrados localmente. Execute "
            "`python scripts/download_hf_assets.py --models` ou defina "
            "`SKIP_HF_DOWNLOAD=1` se já tiver os `.pt` em `hf_models/`."
        )
except Exception as exc:
    st.warning(f"Bootstrap de modelos indisponível: {exc}")

st.set_page_config(
    page_title="Assistente Médico Multimodal",
    page_icon="🩺",
    layout="wide",
)

page_header(
    "Assistente Médico Multimodal",
    "Fluxo guiado para entrada do caso, processamento, resultados e relatório.",
)


show_sidebar_navigation_guide()
show_case_status_sidebar(st.session_state.get("case_id"))

