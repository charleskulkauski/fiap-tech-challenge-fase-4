from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_case_lgpd_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "case_lgpd_context",
    default=None,
)

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "[email_redigido]"),
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[cpf_redigido]"),
    (re.compile(r"\b(?:\+?\d{2}\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b"), "[telefone_redigido]"),
)


class LgpdProcessingBlockedError(RuntimeError):
    """Processamento externo bloqueado por política LGPD ou ausência de consentimento."""


@dataclass(frozen=True)
class LgpdSettings:
    strict_mode: bool
    require_explicit_consent: bool
    allow_azure_openai: bool
    allow_azure_speech: bool
    anonymize_external_payloads: bool
    max_transcript_chars_azure: int
    data_retention_days: int
    processing_purpose: str


def get_lgpd_settings() -> LgpdSettings:
    return LgpdSettings(
        strict_mode=_env_bool("LGPD_STRICT_MODE", default=True),
        require_explicit_consent=_env_bool("LGPD_REQUIRE_EXPLICIT_CONSENT", default=True),
        allow_azure_openai=_env_bool("LGPD_ALLOW_AZURE_OPENAI", default=True),
        allow_azure_speech=_env_bool("LGPD_ALLOW_AZURE_SPEECH", default=True),
        anonymize_external_payloads=_env_bool("LGPD_ANONYMIZE_EXTERNAL_PAYLOADS", default=True),
        max_transcript_chars_azure=max(_env_int("LGPD_MAX_TRANSCRIPT_CHARS_AZURE", 4000), 500),
        data_retention_days=max(_env_int("LGPD_DATA_RETENTION_DAYS", 30), 1),
        processing_purpose=os.environ.get(
            "LGPD_PROCESSING_PURPOSE",
            "triagem_multimodal_apoio_clinico_pesquisa",
        ).strip(),
    )


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def build_lgpd_consent_record(
    *,
    consent_local_processing: bool,
    consent_azure_speech: bool,
    consent_azure_openai: bool,
    purpose: str | None = None,
) -> dict[str, Any]:
    settings = get_lgpd_settings()
    now = datetime.now().isoformat(timespec="seconds")
    expires_at = (
        datetime.now() + timedelta(days=settings.data_retention_days)
    ).isoformat(timespec="seconds")
    return {
        "consent_local_processing": bool(consent_local_processing),
        "consent_azure_speech": bool(consent_azure_speech),
        "consent_azure_openai": bool(consent_azure_openai),
        "purpose": purpose or settings.processing_purpose,
        "consented_at": now,
        "retention_expires_at": expires_at,
        "retention_days": settings.data_retention_days,
        "policy_version": "lgpd-v1",
    }


def extract_lgpd_consent(case_meta: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(case_meta, dict):
        return {}
    lgpd = case_meta.get("lgpd")
    if isinstance(lgpd, dict):
        return lgpd
    return {}


def set_case_lgpd_context(case_meta: dict[str, Any] | None) -> None:
    _case_lgpd_context.set(case_meta if isinstance(case_meta, dict) else None)


def clear_case_lgpd_context() -> None:
    _case_lgpd_context.set(None)


def get_case_lgpd_context() -> dict[str, Any] | None:
    return _case_lgpd_context.get()


def _consent_flags(source: dict[str, Any] | None) -> dict[str, bool]:
    consent = extract_lgpd_consent(source)
    settings = get_lgpd_settings()
    if settings.require_explicit_consent:
        return {
            "local": bool(consent.get("consent_local_processing", False)),
            "speech": bool(consent.get("consent_azure_speech", False)),
            "openai": bool(consent.get("consent_azure_openai", False)),
        }
    return {
        "local": True,
        "speech": settings.allow_azure_speech,
        "openai": settings.allow_azure_openai,
    }


def allows_local_processing(case_meta: dict[str, Any] | None = None) -> bool:
    source = case_meta if case_meta is not None else get_case_lgpd_context()
    return _consent_flags(source)["local"]


def allows_azure_speech(case_meta: dict[str, Any] | None = None) -> bool:
    settings = get_lgpd_settings()
    if not settings.allow_azure_speech:
        return False
    source = case_meta if case_meta is not None else get_case_lgpd_context()
    flags = _consent_flags(source)
    return flags["local"] and flags["speech"]


def allows_azure_openai(case_meta: dict[str, Any] | None = None) -> bool:
    settings = get_lgpd_settings()
    if not settings.allow_azure_openai:
        return False
    source = case_meta if case_meta is not None else get_case_lgpd_context()
    flags = _consent_flags(source)
    return flags["local"] and flags["openai"]


def assert_local_processing_allowed(case_meta: dict[str, Any] | None = None) -> None:
    if allows_local_processing(case_meta):
        return
    raise LgpdProcessingBlockedError(
        "Processamento bloqueado: consentimento local LGPD ausente para este caso."
    )


def assert_azure_speech_allowed() -> None:
    if allows_azure_speech():
        return
    raise LgpdProcessingBlockedError(
        "Transcrição via Azure Speech bloqueada: consentimento LGPD ausente ou "
        "processamento externo desabilitado."
    )


def assert_azure_openai_allowed() -> None:
    if allows_azure_openai():
        return
    raise LgpdProcessingBlockedError(
        "Chamada Azure OpenAI bloqueada: consentimento LGPD ausente ou "
        "processamento externo desabilitado."
    )


def redact_pii(text: str) -> str:
    redacted = text or ""
    for pattern, replacement in _PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def sanitize_transcript_for_external(text: str) -> str:
    settings = get_lgpd_settings()
    cleaned = redact_pii((text or "").strip())
    if not cleaned:
        return ""
    if len(cleaned) > settings.max_transcript_chars_azure:
        cleaned = cleaned[: settings.max_transcript_chars_azure].rstrip() + " […]"
    return cleaned


def sanitize_external_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_lgpd_settings()
    if not settings.anonymize_external_payloads:
        return payload

    blocked_keys = {
        "case_id",
        "observacoes_iniciais",
        "input_files",
        "transcript",
        "transcript_raw",
    }
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if key in blocked_keys:
            continue
        if isinstance(value, str):
            sanitized[key] = redact_pii(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_external_json_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_external_json_payload(item)
                if isinstance(item, dict)
                else redact_pii(str(item))
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def run_local_audio_phase(
    audio_path: Path,
    output_dir: Path,
    consultation_start: datetime,
) -> dict[str, Any]:
    from models.audio_analysis import analyze_audio_risk, save_audio_analysis_with_timestamp

    result: dict[str, Any] = {
        "transcript_path": None,
        "audio_analysis_path": None,
        "audio_analysis": None,
    }
    if not audio_path.is_file():
        return result
    try:
        analysis = analyze_audio_risk(audio_path, "")
        analysis_path = save_audio_analysis_with_timestamp(
            analysis,
            output_dir,
            consultation_start=consultation_start,
        )
        result["audio_analysis"] = analysis
        result["audio_analysis_path"] = analysis_path
    except Exception:
        return result
    return result


def build_lgpd_audit_metadata() -> dict[str, Any]:
    settings = get_lgpd_settings()
    consent = extract_lgpd_consent(get_case_lgpd_context())
    return {
        "strict_mode": settings.strict_mode,
        "require_explicit_consent": settings.require_explicit_consent,
        "allows_azure_speech": allows_azure_speech(),
        "allows_azure_openai": allows_azure_openai(),
        "anonymize_external_payloads": settings.anonymize_external_payloads,
        "processing_purpose": consent.get("purpose") or settings.processing_purpose,
        "retention_expires_at": consent.get("retention_expires_at"),
        "consent_local_processing": bool(consent.get("consent_local_processing", False)),
        "consent_azure_speech": bool(consent.get("consent_azure_speech", False)),
        "consent_azure_openai": bool(consent.get("consent_azure_openai", False)),
    }
