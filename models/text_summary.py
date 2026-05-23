
from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv
from openai import APIError, AzureOpenAI

load_dotenv()

_DEFAULT_API_VERSION: Final[str] = "2024-10-21"

_SYSTEM_PROMPT_PT: Final[str] = (
    "Você é um assistente clínico que resume transcrições de áudio de pacientes. "
    "Produza um resumo objetivo, em português, destacando: (1) queixa principal, "
    "(2) sintomas relatados, (3) intensidade/duração quando mencionadas e "
    "(4) qualquer sinal de alerta. Não invente informações que não estejam no texto."
)

_SYSTEM_PROMPT_EN: Final[str] = (
    "You are a clinical assistant that summarizes patient audio transcriptions. "
    "Produce an objective summary highlighting: (1) chief complaint, (2) reported "
    "symptoms, (3) intensity/duration when mentioned, and (4) any red flags. "
    "Do not invent information that is not present in the text."
)


def _build_client() -> tuple[AzureOpenAI, str]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = (
        os.environ.get("AZURE_OPENAI_API_VERSION", "").strip() or _DEFAULT_API_VERSION
    )

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Variáveis de ambiente ausentes para Azure OpenAI: "
            + ", ".join(missing)
            + ". Configure no arquivo .env."
        )

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return client, deployment


def summarize_text(
    text: str,
    *,
    language: str = "pt",
    max_words: int | None = 120,
    temperature: float = 0.2,
) -> str:
    from models.lgpd_compliance import (
        assert_azure_openai_allowed,
        sanitize_transcript_for_external,
    )

    assert_azure_openai_allowed()
    if not text or not text.strip():
        return ""

    client, deployment = _build_client()

    system_prompt = _SYSTEM_PROMPT_EN if language.lower().startswith("en") else _SYSTEM_PROMPT_PT
    length_hint = (
        f" Limite o resumo a aproximadamente {max_words} palavras."
        if max_words and max_words > 0
        else ""
    )
    safe_text = sanitize_transcript_for_external(text)

    user_prompt = (
        "Resuma a transcrição a seguir mantendo apenas o conteúdo presente nela."
        f"{length_hint}\n\n--- TRANSCRIÇÃO ---\n{safe_text}"
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIError as exc:
        raise RuntimeError(
            f"Falha ao chamar Azure OpenAI ({deployment}): {exc}"
        ) from exc

    if not response.choices:
        return ""

    summary = (response.choices[0].message.content or "").strip()
    return summary
