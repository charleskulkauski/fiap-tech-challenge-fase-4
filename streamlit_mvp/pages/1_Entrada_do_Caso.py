from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

import streamlit as st

                                                                     
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_mvp.utils.case_store import ensure_case_dirs, save_case_meta, save_case_status
from streamlit_mvp.utils.case_policy import CASE_CATEGORIES, normalize_case_category
from streamlit_mvp.utils.project import normalize_case_id
from models.lgpd_compliance import build_lgpd_consent_record, get_lgpd_settings
from streamlit_mvp.utils.ui import (
    page_header,
    show_case_status_sidebar,
    show_sidebar_navigation_guide,
)
from streamlit_mvp.utils.validators import (
    AUDIO_EXTENSIONS,
    MAX_AUDIO_MB,
    MAX_VIDEO_MB,
    VIDEO_EXTENSIONS,
    validate_uploaded_file,
)

st.set_page_config(page_title="Entrada do Caso", page_icon="🗂️", layout="wide")

page_header(
    "🗂️ Entrada do Caso",
    "Abrir caso clínico, validar uploads e iniciar o fluxo multimodal.",
)
show_sidebar_navigation_guide()
show_case_status_sidebar(st.session_state.get("case_id"))

with st.form("form_entrada"):
    c1, c2 = st.columns(2)
    with c1:
        raw_case_id = st.text_input(
            "Case ID",
            placeholder="ex.: caso-gestante-001",
            help="Use um identificador sem dados pessoais diretos.",
        )
    with c2:
        categoria_caso = st.selectbox(
            "Categoria do caso (obrigatória)",
            list(CASE_CATEGORIES),
            help=(
                "O processamento será estritamente orientado por essa categoria, "
                "incluindo regras de proibição por modalidade."
            ),
        )

    observacoes = st.text_area(
        "Observações iniciais",
        placeholder="Informações clínicas iniciais para rastreabilidade do caso.",
        height=120,
    )

    video_file = st.file_uploader(
        "Upload do vídeo clínico",
        type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)],
    )
    audio_file = st.file_uploader(
        "Upload do áudio da consulta",
        type=[ext.lstrip(".") for ext in sorted(AUDIO_EXTENSIONS)],
    )

    st.caption("Envie pelo menos um insumo: vídeo clínico, áudio da consulta ou ambos.")

    lgpd_settings = get_lgpd_settings()
    st.divider()
    st.subheader("Consentimento LGPD")
    st.caption(
        "Conforme a Lei 13.709/2018, o processamento exige consentimento explícito. "
        "Serviços Microsoft/Azure são opcionais e só são usados quando autorizados."
    )

    consent_local = st.checkbox(
        "Autorizo o processamento local dos dados deste caso para triagem multimodal "
        "(obrigatório).",
        value=False,
    )
    consent_speech = st.checkbox(
        "Autorizo o envio do áudio para transcrição via Microsoft Azure Speech (opcional).",
        value=False,
        disabled=not lgpd_settings.allow_azure_speech,
    )
    consent_openai = st.checkbox(
        "Autorizo o envio de dados minimizados para enriquecimento via "
        "Microsoft Azure OpenAI (opcional).",
        value=False,
        disabled=not lgpd_settings.allow_azure_openai,
    )

    submit = st.form_submit_button("Criar caso e iniciar análise")
    st.divider()

if submit:
    errors: list[str] = []
    case_id = normalize_case_id(raw_case_id)
    if not case_id:
        errors.append("Informe um case_id válido.")

    if video_file is None and audio_file is None:
        errors.append("Envie ao menos um arquivo: vídeo ou áudio.")

    video_bytes = b""
    audio_bytes = b""
    if video_file is not None:
        video_bytes = video_file.getvalue()
        err = validate_uploaded_file(
            video_file.name,
            video_bytes,
            VIDEO_EXTENSIONS,
            MAX_VIDEO_MB,
        )
        if err:
            errors.append(f"Vídeo: {err}")
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        err = validate_uploaded_file(
            audio_file.name,
            audio_bytes,
            AUDIO_EXTENSIONS,
            MAX_AUDIO_MB,
        )
        if err:
            errors.append(f"Áudio: {err}")

    if lgpd_settings.require_explicit_consent and not consent_local:
        errors.append(
            "É necessário autorizar o processamento local do caso (LGPD) para continuar."
        )

    if errors:
        for msg in errors:
            st.error(msg)
    else:
        files = ensure_case_dirs(case_id)

        if video_file is not None:
            files.video_input.write_bytes(video_bytes)
        else:
            files.video_input.unlink(missing_ok=True)

        audio_target: Path | None = None
        if audio_file is not None:
            audio_suffix = Path(audio_file.name).suffix.lower() or ".wav"
            audio_target = files.inputs_dir / f"audio_input{audio_suffix}"
            for old_audio in files.inputs_dir.glob("audio_input.*"):
                if old_audio != audio_target:
                    old_audio.unlink(missing_ok=True)
            audio_target.write_bytes(audio_bytes)
        else:
            for old_audio in files.inputs_dir.glob("audio_input.*"):
                old_audio.unlink(missing_ok=True)

        meta = {
            "case_id": case_id,
            "categoria_caso": normalize_case_category(categoria_caso),
            "observacoes_iniciais": observacoes.strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_files": {
                "video_name": video_file.name if video_file is not None else None,
                "audio_name": audio_file.name if audio_file is not None else None,
                "audio_stored_as": audio_target.name if audio_target is not None else None,
                "has_video": video_file is not None,
                "has_audio": audio_file is not None,
            },
            "lgpd": build_lgpd_consent_record(
                consent_local_processing=consent_local,
                consent_azure_speech=consent_speech and lgpd_settings.allow_azure_speech,
                consent_azure_openai=consent_openai and lgpd_settings.allow_azure_openai,
            ),
        }
        save_case_meta(case_id, meta)
        save_case_status(
            case_id,
            {
                "case_created": True,
                "processing_done": False,
                "report_ready": False,
                "case_closed": False,
                "last_error": "",
                "step": "created",
            },
        )

        st.session_state["case_id"] = case_id
        st.success(f"Caso `{case_id}` criado com sucesso.")
        st.switch_page("pages/2_Processamento_e_Status.py")
