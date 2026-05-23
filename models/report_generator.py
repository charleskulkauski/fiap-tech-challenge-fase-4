
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Final, Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, AzureOpenAI

load_dotenv()


_DEFAULT_API_VERSION: Final[str] = "2024-10-21"
_DEFAULT_REPORT_MODE: Final[str] = "local"

                                                                     
                                    
                                              
PAIN_MODERATE: Final[float] = 25.0
PAIN_SEVERE: Final[float] = 40.0

                                                                          
                                           
PEAK_MIN_FRAMES: Final[int] = 3
                                                                    
MAX_PEAKS_IN_PROMPT: Final[int] = 8
                                               
MAX_INSTRUMENTS_IN_PROMPT: Final[int] = 10
                                                                            
MAX_TRANSCRIPT_CHARS: Final[int] = 8000


_SYSTEM_PROMPT: Final[str] = (
    "Você é um assistente clínico de apoio à equipe médica especializada em "
    "saúde da mulher. Você NÃO produz diagnósticos: produz um resumo "
    "estruturado para triagem não diagnóstica, baseado APENAS nos dados "
    "fornecidos (transcrição de áudio + métricas agregadas de vídeo). "
    "Quando algum dado não estiver presente, explicite a ausência em vez "
    "de inferir. Escreva em português, de forma objetiva e técnica."
)

_USER_PROMPT_TEMPLATE_MULTIMODAL: Final[str] = """Dados da consulta a serem analisados:

== TRANSCRIÇÃO DO ÁUDIO ==
{transcript}

== MÉTRICAS AGREGADAS DO VÍDEO ==
{video_summary}

Com base SOMENTE nos dados acima, gere um relatório em Markdown com a \
seguinte estrutura, exatamente nesta ordem:

# Relatório clínico de apoio (não diagnóstico)

## 1. Resumo da consulta
Resuma em até 6 linhas o conteúdo da transcrição (queixas, sintomas, \
duração mencionada). Se a transcrição estiver vazia, registre \
"Transcrição não disponível.".

## 2. Sinais de desconforto (proxy de vídeo)
Comente os indicadores agregados de dor/desconforto extraídos do vídeo \
(`pain_proxy` médio, `pain_smoothed` máximo, picos sustentados, flags de \
postura). Deixe claro que se trata de um proxy não clínico.

## 3. Instrumentos detectados
Liste os instrumentos / objetos que apareceram no vídeo com sua \
frequência (em frames). Se a lista vier vazia, registre "Nenhum \
instrumento relevante detectado.".

## 4. Recomendação de triagem (não diagnóstica)
Sugira encaminhamentos de triagem proporcionais ao quadro observado \
(ex.: revisão da consulta pela equipe, acompanhamento psicológico, \
avaliação ginecológica direcionada, reforço de orientações). Use tom \
prudente; reforce que NÃO é um diagnóstico.

## 5. Limitações
Cite limitações dos dados (transcrição automática, score proxy, modelo \
de detecção genérico se for o caso) em até 3 linhas.
"""

_USER_PROMPT_TEMPLATE_VIDEO_ONLY: Final[str] = """Dados da consulta a serem analisados:

== MÉTRICAS AGREGADAS DO VÍDEO ==
{video_summary}

Com base SOMENTE nos dados acima, gere um relatório em Markdown com a \
seguinte estrutura, exatamente nesta ordem:

# Relatório clínico de apoio (não diagnóstico)

## 1. Resumo dos eventos observados no vídeo
Explique em até 6 linhas o que está acontecendo na consulta com base nos \
indicadores visuais, destacando padrões de desconforto, comportamento e \
eventos relevantes ao longo do tempo.

## 2. Frames/momentos críticos indicados
Liste os principais intervalos críticos com referência temporal (ex.: \
t=12.0s-18.0s), justificando por que merecem revisão clínica.

## 3. Instrumentos/objetos detectados
Liste os instrumentos / objetos detectados e frequência (frames). Se não \
houver, registre "Nenhum instrumento relevante detectado.".

## 4. Recomendação de triagem baseada em vídeo (não diagnóstica)
Sugira encaminhamentos prudentes focados no que foi observado visualmente, \
incluindo priorização de revisão humana dos momentos críticos.

## 5. Limitações
Cite limitações dos dados visuais em até 3 linhas (proxy de dor/postura, \
detecção automática e ausência de áudio/transcrição).
"""

_USER_PROMPT_TEMPLATE_LAPARO_VIDEO_ONLY: Final[str] = """Dados da consulta a serem analisados:

== MÉTRICAS AGREGADAS DO VÍDEO ==
{video_summary}

Com base SOMENTE nos dados acima, gere um relatório em Markdown com a \
seguinte estrutura, exatamente nesta ordem:

# Relatório clínico de apoio (não diagnóstico)

## 1. Resumo dos achados laparoscópicos
Resuma em até 6 linhas os principais achados visuais do procedimento interno, \
sem usar linguagem de dor postural/comportamental.

## 2. Evidências de sangramento no vídeo
Descreva objetivamente se há ou não evidências de sangramento no vídeo e o \
nível de evidência disponível nas métricas agregadas.

## 3. Instrumentos detectados
Informe a quantidade de instrumentos/objetos detectados (frames com detecção) \
e liste os nomes/classes detectados com frequência. Se não houver, registre \
"Nenhum instrumento relevante detectado.".

## 4. Recomendação de triagem baseada em vídeo (não diagnóstica)
Produza recomendação focada APENAS em revisão técnica laparoscópica \
(instrumentação, sangramento e validação de momentos críticos), sem \
recomendações de acolhimento psicológico, dor postural ou comportamento.

## 5. Limitações
Cite limitações em até 3 linhas para laparoscopia (detecção automática de \
objetos/sangramento, ausência de confirmação clínica direta e possível erro de \
classificação).
"""

_USER_PROMPT_TEMPLATE_AUDIO_ONLY: Final[str] = """Dados da consulta a serem analisados:

== TRANSCRIÇÃO DO ÁUDIO ==
{transcript}

Com base SOMENTE nos dados acima, gere um relatório em Markdown com a \
seguinte estrutura, exatamente nesta ordem:

# Relatório clínico de apoio (não diagnóstico)

## 1. Resumo da consulta (transcrição)
Resuma em até 8 linhas os principais relatos, queixas, sintomas e contexto \
trazidos na fala. Se a transcrição estiver vazia, registre \
"Transcrição não disponível.".

## 2. Pontos de atenção linguísticos/comportamentais
Destaque termos, padrões de fala e sinais textuais relevantes para triagem \
não diagnóstica (ex.: medo, ansiedade, dor, urgência, violência).

## 3. Plano de acompanhamento pós-áudio
Sugira próximos passos de acompanhamento após esta etapa de áudio \
(monitoramento, retorno, acolhimento, revisão multiprofissional), sem \
diagnóstico.

## 4. Recomendação de triagem (não diagnóstica)
Produza recomendação final objetiva e prudente, reforçando limites do uso \
exclusivo da transcrição.

## 5. Limitações
Cite limitações dos dados em até 3 linhas (transcrição automática, ausência \
de vídeo e ausência de exame clínico presencial).
"""


@dataclass
class VideoAggregateSummary:

    frames_total: int = 0
    duration_sec: float = 0.0
    pain_proxy_mean: float | None = None
    pain_proxy_max: float | None = None
    pain_smoothed_mean: float | None = None
    pain_smoothed_max: float | None = None
    pain_moderate_frames: int = 0
    pain_severe_frames: int = 0
    pain_peaks: list[dict[str, float]] = field(default_factory=list)
    behavior_score_mean: float | None = None
    flag_rates: dict[str, float] = field(default_factory=dict)
    instrument_counts: dict[str, int] = field(default_factory=dict)
    n_frames_with_instrument: int = 0

    def to_prompt_block(self) -> str:
        lines: list[str] = []
        lines.append(
            f"- Frames analisados: {self.frames_total}"
            f" ({self.duration_sec:.1f}s de vídeo)"
        )

        if self.pain_proxy_mean is not None:
            lines.append(
                "- pain_proxy (DeepFace, %): "
                f"média={self.pain_proxy_mean:.2f}, "
                f"máx={self.pain_proxy_max:.2f}"
            )
        else:
            lines.append("- pain_proxy: sem rostos detectados ao longo do vídeo")

        if self.pain_smoothed_mean is not None:
            lines.append(
                "- pain_smoothed (média móvel, %): "
                f"média={self.pain_smoothed_mean:.2f}, "
                f"máx={self.pain_smoothed_max:.2f}"
            )

        lines.append(
            f"- Frames com desconforto moderado (>= {PAIN_MODERATE:.0f}%): "
            f"{self.pain_moderate_frames}"
        )
        lines.append(
            f"- Frames com dor severa (>= {PAIN_SEVERE:.0f}%): "
            f"{self.pain_severe_frames}"
        )

        if self.pain_peaks:
            lines.append(
                f"- Picos sustentados (>={PAIN_MODERATE:.0f}% por "
                f">={PEAK_MIN_FRAMES} frames):"
            )
            for peak in self.pain_peaks[:MAX_PEAKS_IN_PROMPT]:
                lines.append(
                    f"    * t={peak['start_sec']:.1f}s–"
                    f"{peak['end_sec']:.1f}s "
                    f"(máx={peak['peak_value']:.1f}%)"
                )
        else:
            lines.append("- Picos sustentados: nenhum")

        if self.behavior_score_mean is not None:
            lines.append(
                f"- behavior_score médio (proxy postural): "
                f"{self.behavior_score_mean:.2f}"
            )

        if self.flag_rates:
            flag_parts = ", ".join(
                f"{name}={rate * 100:.0f}%"
                for name, rate in sorted(self.flag_rates.items())
            )
            lines.append(f"- Flags de postura (% de frames ativos): {flag_parts}")

        if self.instrument_counts:
            top = sorted(
                self.instrument_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:MAX_INSTRUMENTS_IN_PROMPT]
            inst_parts = ", ".join(f"{name} ({count})" for name, count in top)
            lines.append(
                "- Instrumentos/objetos detectados (frames com pelo menos "
                f"1 detecção): {self.n_frames_with_instrument}"
            )
            lines.append(f"- Top classes detectadas: {inst_parts}")
        else:
            lines.append("- Instrumentos/objetos detectados: nenhum relevante")

        return "\n".join(lines)


def _detect_pain_peaks(
    timestamps: pd.Series,
    pain_smoothed: pd.Series,
    *,
    threshold: float = PAIN_MODERATE,
    min_frames: int = PEAK_MIN_FRAMES,
) -> list[dict[str, float]]:
    if pain_smoothed.empty:
        return []

    mask = pain_smoothed >= threshold
    peaks: list[dict[str, float]] = []

    in_peak = False
    start_idx = 0
    peak_value = 0.0
    for i, (is_high, value) in enumerate(zip(mask, pain_smoothed)):
        if is_high and not in_peak:
            in_peak = True
            start_idx = i
            peak_value = float(value)
        elif is_high and in_peak:
            if float(value) > peak_value:
                peak_value = float(value)
        elif not is_high and in_peak:
            length = i - start_idx
            if length >= min_frames:
                peaks.append(
                    {
                        "start_sec": float(timestamps.iloc[start_idx]),
                        "end_sec": float(timestamps.iloc[i - 1]),
                        "peak_value": peak_value,
                        "length_frames": float(length),
                    }
                )
            in_peak = False
            peak_value = 0.0

    if in_peak:
        length = len(mask) - start_idx
        if length >= min_frames:
            peaks.append(
                {
                    "start_sec": float(timestamps.iloc[start_idx]),
                    "end_sec": float(timestamps.iloc[-1]),
                    "peak_value": peak_value,
                    "length_frames": float(length),
                }
            )

    peaks.sort(key=lambda p: p["peak_value"], reverse=True)
    return peaks


def _aggregate_instruments(
    df: pd.DataFrame,
    instruments_df: pd.DataFrame | None,
) -> tuple[dict[str, int], int]:
    if instruments_df is not None and not instruments_df.empty:
        class_series = (
            instruments_df["class_name"]
            .astype(str)
            .str.strip()
        )
        class_series = class_series[class_series != ""]
        counts = class_series.value_counts().to_dict()
        n_frames = int(instruments_df["frame_idx"].nunique())
        return counts, n_frames

    if "instruments_seen" not in df.columns:
        return {}, 0

    counts: dict[str, int] = {}
    n_frames = 0
    for raw in df["instruments_seen"].fillna(""):
        if not raw:
            continue
        n_frames += 1
        for name in str(raw).split("|"):
            name = name.strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
    return counts, n_frames


_EVENTS_CSV_PT_TO_EN: dict[str, str] = {
    "proxy_dor": "pain_proxy",
    "dor_suavizada_pct": "pain_smoothed",
    "score_postural": "behavior_score",
}


def _normalize_events_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        pt: en
        for pt, en in _EVENTS_CSV_PT_TO_EN.items()
        if pt in df.columns and en not in df.columns
    }
    if not rename:
        return df
    return df.rename(columns=rename)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    raw = df[column]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    converted = pd.to_numeric(raw, errors="coerce")
    if isinstance(converted, pd.Series):
        return converted
    return pd.Series([converted], dtype="float64")


def _clean_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return _numeric_column(df, column).dropna()


def aggregate_video_events(
    events_csv_path: str | Path,
    instruments_csv_path: str | Path | None = None,
) -> VideoAggregateSummary:
    events_csv_path = Path(events_csv_path)
    df = _normalize_events_df_columns(pd.read_csv(events_csv_path))

    instruments_df: pd.DataFrame | None = None
    if instruments_csv_path is not None:
        p = Path(instruments_csv_path)
        if p.is_file():
            try:
                instruments_df = pd.read_csv(p)
            except Exception:
                instruments_df = None

    summary = VideoAggregateSummary()
    summary.frames_total = int(len(df))
    if "timestamp_sec" in df.columns and not df.empty:
        summary.duration_sec = float(df["timestamp_sec"].max())

    pain_proxy = _clean_numeric_column(df, "pain_proxy")
    if not pain_proxy.empty:
        summary.pain_proxy_mean = float(pain_proxy.mean())
        summary.pain_proxy_max = float(pain_proxy.max())

    pain_smoothed = _numeric_column(df, "pain_smoothed")
    pain_smoothed_valid = _clean_numeric_column(df, "pain_smoothed")
    if not pain_smoothed_valid.empty:
        ps_clean = pain_smoothed.fillna(0.0)
        summary.pain_smoothed_mean = float(ps_clean.mean())
        summary.pain_smoothed_max = float(ps_clean.max())
        summary.pain_moderate_frames = int((ps_clean >= PAIN_MODERATE).sum())
        summary.pain_severe_frames = int((ps_clean >= PAIN_SEVERE).sum())
        if "timestamp_sec" in df.columns:
            summary.pain_peaks = _detect_pain_peaks(
                df["timestamp_sec"], ps_clean
            )

    behavior = _clean_numeric_column(df, "behavior_score")
    if not behavior.empty:
        summary.behavior_score_mean = float(behavior.mean())

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols and summary.frames_total > 0:
        flag_rates: dict[str, float] = {}
        for col in flag_cols:
            try:
                rate = float(df[col].astype(float).mean())
            except Exception:
                continue
            flag_rates[col.replace("flag_", "")] = rate
        summary.flag_rates = flag_rates

    counts, n_frames_with_instr = _aggregate_instruments(df, instruments_df)
    summary.instrument_counts = counts
    summary.n_frames_with_instrument = n_frames_with_instr

    return summary


def _load_transcript(transcript_path: str | Path) -> str:
    from models.lgpd_compliance import sanitize_transcript_for_external

    path = Path(transcript_path)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    text = sanitize_transcript_for_external(text)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS].rstrip() + " […]"
    return text


def _report_modality(
    transcript: str,
    summary: VideoAggregateSummary | None,
) -> str:
    has_audio = bool((transcript or "").strip())
    has_video = summary is not None and summary.frames_total > 0
    if has_audio and has_video:
        return "multimodal"
    if has_video:
        return "video_only"
    return "audio_only"


def _normalize_report_profile(raw_profile: str | None) -> str:
    normalized = (raw_profile or "").strip().lower()
    if "laparo" in normalized or "gineco" in normalized or "cirurg" in normalized:
        return "laparo"
    return "default"


def _infer_report_profile(
    transcript: str,
    summary: VideoAggregateSummary | None,
    explicit_profile: str | None,
) -> str:
    normalized = _normalize_report_profile(explicit_profile)
    if normalized != "default":
        return normalized
    if summary is None:
        return "default"
    has_instruments = bool(summary.instrument_counts)
    low_patient_signal = (
        (summary.pain_proxy_mean is None or summary.pain_proxy_mean <= 1.0)
        and (summary.pain_smoothed_max is None or summary.pain_smoothed_max <= 1.0)
        and (summary.behavior_score_mean is None or summary.behavior_score_mean <= 1.0)
    )
    if has_instruments and low_patient_signal and not (transcript or "").strip():
        return "laparo"
    return "default"


def build_prompt(
    transcript: str,
    summary: VideoAggregateSummary | None,
    *,
    report_profile: str | None = None,
) -> str:
    transcript_block = transcript.strip() or "(transcrição vazia ou não disponível)"
    video_block = (
        summary.to_prompt_block()
        if summary is not None
        else "- Métricas de vídeo indisponíveis."
    )

    profile = _infer_report_profile(transcript, summary, report_profile)
    modality = _report_modality(transcript, summary)
    if profile == "laparo" and modality in {"video_only", "multimodal"}:
        return _USER_PROMPT_TEMPLATE_LAPARO_VIDEO_ONLY.format(video_summary=video_block)
    if modality == "multimodal":
        return _USER_PROMPT_TEMPLATE_MULTIMODAL.format(
            transcript=transcript_block,
            video_summary=video_block,
        )
    if modality == "video_only":
        return _USER_PROMPT_TEMPLATE_VIDEO_ONLY.format(video_summary=video_block)
    return _USER_PROMPT_TEMPLATE_AUDIO_ONLY.format(transcript=transcript_block)


def _build_azure_client() -> tuple[AzureOpenAI, str]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = (
        os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
        or _DEFAULT_API_VERSION
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


def call_llm(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    system_prompt: str | None = None,
) -> str:
    from models.lgpd_compliance import assert_azure_openai_allowed

    assert_azure_openai_allowed()
    client, deployment = _build_azure_client()
    try:
        response = client.chat.completions.create(
            model=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except APIError as exc:
        raise RuntimeError(
            f"Falha ao chamar Azure OpenAI ({deployment}): {exc}"
        ) from exc

    if not response.choices:
        return ""
    content = response.choices[0].message.content or ""
    return content.strip()


def _report_mode() -> str:
    raw = os.environ.get("REPORT_MODE", _DEFAULT_REPORT_MODE).strip().lower()
    if raw not in {"local", "hybrid", "azure"}:
        return _DEFAULT_REPORT_MODE
    return raw


def _build_local_fallback_llm_markdown(
    transcript: str,
    summary: VideoAggregateSummary | None,
    *,
    report_profile: str | None = None,
) -> str:
    transcript_clean = (transcript or "").strip()
    profile = _infer_report_profile(transcript_clean, summary, report_profile)
    if transcript_clean:
        resumo_consulta = (
            f"Transcrição disponível com {len(transcript_clean)} caracteres. "
            "Há conteúdo textual suficiente para triagem inicial, com necessidade "
            "de revisão humana detalhada. Este resumo foi gerado localmente, "
            "sem uso de Azure OpenAI."
        )
    else:
        resumo_consulta = (
            "Transcrição não disponível. Este resumo foi gerado localmente, "
            "sem uso de Azure OpenAI."
        )

    modality = _report_modality(transcript_clean, summary)
    pain_mean = (
        summary.pain_proxy_mean
        if summary is not None and summary.pain_proxy_mean is not None
        else 0.0
    )
    pain_max = (
        summary.pain_smoothed_max
        if summary is not None and summary.pain_smoothed_max is not None
        else 0.0
    )
    if pain_max >= PAIN_SEVERE:
        pain_band = "desconforto elevado (proxy)"
    elif pain_max >= PAIN_MODERATE:
        pain_band = "desconforto moderado (proxy)"
    else:
        pain_band = "desconforto baixo/estável (proxy)"
    sinais = (
        "Os indicadores de vídeo sugerem "
        f"{pain_band}, com média do proxy de dor em {pain_mean:.2f}% "
        f"e pico do score suavizado em {pain_max:.2f}%. "
        "Esses sinais são não clínicos e devem ser interpretados com cautela."
    )

    if summary is not None and summary.instrument_counts:
        top = sorted(
            summary.instrument_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:MAX_INSTRUMENTS_IN_PROMPT]
        instrumentos = "; ".join(f"{name} ({count} frames)" for name, count in top)
    else:
        instrumentos = "Nenhum instrumento relevante detectado."

    lower_transcript = transcript_clean.lower()
    has_abuse_context = any(
        token in lower_transcript
        for token in ("abuso", "agress", "violên", "violenc", "ameaça", "ameaca")
    )
    has_anxiety_context = any(
        token in lower_transcript
        for token in ("ansied", "medo", "pânico", "panico", "nervos")
    )

    context_tags: list[str] = []
    if has_abuse_context:
        context_tags.append("acolhimento/abuso")
    if has_anxiety_context:
        context_tags.append("ansiedade")
    if pain_max >= PAIN_SEVERE:
        context_tags.append("desconforto elevado")
    elif pain_max >= PAIN_MODERATE:
        context_tags.append("desconforto moderado")
    else:
        context_tags.append("desconforto baixo")
    context_tags_text = ", ".join(context_tags)

    recomendacao_prompt = (
        "Você é um assistente de triagem clínica não diagnóstica. "
        "Com base apenas nos parâmetros abaixo, escreva 4 a 6 linhas de recomendação "
        "objetiva para orientação inicial da equipe e do paciente, em português, "
        "sem diagnóstico e com linguagem prudente.\n\n"
        f"Parâmetros: {context_tags_text}; "
        f"pain_proxy_medio={pain_mean:.2f}%; pain_smoothed_max={pain_max:.2f}%."
    )
    try:
        recomendacao = call_llm(
            recomendacao_prompt,
            temperature=0.1,
            max_tokens=260,
        )
    except RuntimeError:
        recomendacao = (
            "Recomenda-se acolhimento imediato em ambiente seguro, escuta qualificada "
            "e avaliação multiprofissional prioritária. Considerar suporte em saúde "
            "mental para sintomas de ansiedade e monitoramento clínico do desconforto. "
            "Em caso de risco atual, orientar acesso à rede de proteção e serviços "
            "de urgência. Este conteúdo é apoio de triagem e não substitui avaliação médica."
        )

    limitacoes = (
        "A transcrição pode conter erros de reconhecimento; os indicadores de dor/"
        "postura e detecção visual são proxies algorítmicos sujeitos a falso positivo/"
        "negativo; interpretação final deve ser feita por profissional habilitado."
    )

    if profile == "laparo" and summary is not None:
        blood_count = sum(
            int(count)
            for name, count in summary.instrument_counts.items()
            if any(token in str(name).lower() for token in ("bleeding", "sangramento", "blood"))
        )
        if blood_count > 0:
            blood_text = (
                f"Foram observadas evidências de sangramento nas detecções automáticas "
                f"(ocorrências agregadas: {blood_count}). Recomenda-se revisão humana "
                "dos trechos correspondentes."
            )
        else:
            blood_text = (
                "Não foram observadas evidências explícitas de sangramento nas classes "
                "detectadas automaticamente neste processamento."
            )
        total_instruments = int(summary.n_frames_with_instrument)
        laparo_recomendacao = (
            "Priorizar revisão técnica dos momentos com maior densidade de detecção, "
            "confirmando instrumentação utilizada e consistência temporal do procedimento. "
            "Se houver suspeita clínica de sangramento, considerar nova inferência com "
            "modelo especializado e validação manual dos frames críticos."
        )
        laparo_limitacoes = (
            "Relatório baseado em detecção automática de objetos/eventos em vídeo, "
            "sujeita a falso positivo/negativo e dependente da qualidade do frame. "
            "Não substitui confirmação clínica direta intraoperatória."
        )
        return (
            "# Relatório clínico de apoio (não diagnóstico)\n\n"
            "### 1. Resumo dos achados laparoscópicos\n"
            f"Foram analisados {summary.frames_total} frames ({summary.duration_sec:.1f}s), "
            f"com detecção de instrumentos em {total_instruments} frames. "
            "A análise foi focada em instrumentação e sinais visuais internos do procedimento.\n\n"
            "### 2. Evidências de sangramento no vídeo\n"
            f"{blood_text}\n\n"
            "### 3. Instrumentos detectados\n"
            f"{instrumentos}\n\n"
            "### 4. Recomendação de triagem baseada em vídeo (não diagnóstica)\n"
            f"{laparo_recomendacao}\n\n"
            "### 5. Limitações deste assistente\n"
            f"{laparo_limitacoes}\n"
        )

    if modality == "multimodal":
        return (
            "# Relatório clínico de apoio (não diagnóstico)\n\n"
            "## 1. Resumo da consulta\n"
            f"{resumo_consulta}\n\n"
            "## 2. Sinais de desconforto (proxy de vídeo)\n"
            f"{sinais}\n\n"
            "## 3. Instrumentos detectados\n"
            f"{instrumentos}\n\n"
            "## 4. Recomendação de triagem (não diagnóstica)\n"
            f"Parâmetros de triagem utilizados: {context_tags_text}; "
            f"pain_proxy médio={pain_mean:.2f}%; pain_smoothed máximo={pain_max:.2f}%.\n\n"
            f"{recomendacao}\n\n"
            "## 5. Limitações\n"
            f"{limitacoes}\n"
        )

    if modality == "video_only":
        momentos = "Sem picos críticos sustentados registrados."
        if summary is not None and summary.pain_peaks:
            momentos = "; ".join(
                f"t={peak['start_sec']:.1f}s-{peak['end_sec']:.1f}s (pico {peak['peak_value']:.1f}%)"
                for peak in summary.pain_peaks[:5]
            )
        return (
            "# Relatório clínico de apoio (não diagnóstico)\n\n"
            "### 1. Resumo dos eventos observados no vídeo\n"
            f"{sinais}\n\n"
            "### 2. Frames/momentos críticos indicados\n"
            f"{momentos}\n\n"
            "### 3. Instrumentos/objetos detectados\n"
            f"{instrumentos}\n\n"
            "### 4. Recomendação de triagem baseada em vídeo (não diagnóstica)\n"
            f"{recomendacao}\n\n"
            "### 5. Limitações deste assistente\n"
            "Análise baseada apenas em vídeo e proxies algorítmicos; sem confirmação "
            "diagnóstica e sem evidências da fala da paciente.\n"
        )

    return (
        "# Relatório clínico de apoio (não diagnóstico)\n\n"
        "### 1. Resumo da consulta (transcrição)\n"
        f"{resumo_consulta}\n\n"
        "### 2. Pontos de atenção linguísticos/comportamentais\n"
        f"Contexto textual identificado: {context_tags_text}.\n\n"
        "### 3. Plano de acompanhamento pós-áudio\n"
        f"{recomendacao}\n\n"
        "### 4. Recomendação de triagem (não diagnóstica)\n"
        "Reforçar monitoramento clínico e retorno assistencial baseado nos relatos verbais, "
        "sem inferência diagnóstica.\n\n"
        "### 5. Limitações deste assistente\n"
        "Análise baseada apenas na transcrição automática de áudio, sem sinais visuais "
        "do exame e sujeita a perdas de contexto.\n"
    )


def _build_markdown_report(
    *,
    llm_markdown: str,
    transcript: str,
    summary: VideoAggregateSummary | None,
    transcript_path: Path | None,
    events_csv_path: Path | None,
    instruments_csv_path: Path | None,
    generated_at: datetime,
    report_profile: str | None = None,
) -> str:
    profile = _infer_report_profile(transcript, summary, report_profile)
    header_lines = [
        "**Relatório gerado automaticamente**",
        f"**Data de geração:** {generated_at.isoformat(timespec='seconds')}",
        "",
        "*Apoio à triagem clínica — não constitui diagnóstico médico.*",
        "",
    ]
    metadata_lines = [
        "## Metadados",
        "",
        f"- Gerado em: `{generated_at.isoformat(timespec='seconds')}`",
        f"- Transcript: `{transcript_path}`"
        if transcript_path
        else "- Transcript: (não informado)",
        (
            f"- video_events.csv: `{events_csv_path}`"
            if events_csv_path
            else "- video_events.csv: (não informado)"
        ),
        (
            f"- video_instrument_events.csv: `{instruments_csv_path}`"
            if instruments_csv_path
            else "- video_instrument_events.csv: (não informado)"
        ),
        "",
    ]

    if profile == "laparo" and summary is not None:
        blood_rows = sorted(
            (
                (name, int(count))
                for name, count in summary.instrument_counts.items()
                if any(
                    token in str(name).lower()
                    for token in ("bleeding", "sangramento", "blood")
                )
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        instrument_rows = sorted(
            (
                (name, int(count))
                for name, count in summary.instrument_counts.items()
                if not any(
                    token in str(name).lower()
                    for token in ("bleeding", "sangramento", "blood")
                )
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )[:MAX_INSTRUMENTS_IN_PROMPT]
        laparo_lines: list[str] = [
            f"- Frames analisados: {summary.frames_total} ({summary.duration_sec:.1f}s de vídeo)",
            f"- Frames com instrumentos detectados: {summary.n_frames_with_instrument}",
        ]
        if instrument_rows:
            laparo_lines.append(
                "- Top classes de instrumentos: "
                + ", ".join(f"{name} ({count})" for name, count in instrument_rows)
            )
        else:
            laparo_lines.append("- Top classes de instrumentos: nenhuma detectada")
        if blood_rows:
            laparo_lines.append(
                "- Classes de sangramento detectadas: "
                + ", ".join(f"{name} ({count})" for name, count in blood_rows)
            )
        else:
            laparo_lines.append("- Classes de sangramento detectadas: nenhuma")
        appendix_metric_block = "\n".join(laparo_lines)
    else:
        appendix_metric_block = (
            summary.to_prompt_block()
            if summary is not None
            else "(métricas de vídeo indisponíveis)"
        )

    appendix_title = (
        "## Apêndice — Auditoria laparoscópica (instrumentos e sangramento)"
        if profile == "laparo"
        else "## Apêndice — Métricas brutas (auditoria)"
    )

    appendix_lines = [
        "",
        "---",
        "",
        appendix_title,
        "",
        "```",
        appendix_metric_block,
        "```",
        "",
    ]
    if transcript.strip():
        appendix_lines.extend(
            [
                "### Trecho da transcrição utilizada",
                "",
                "```",
                transcript[:1500] + (" […]" if len(transcript) > 1500 else ""),
                "```",
                "",
            ]
        )

    parts: list[str] = []
    parts.extend(header_lines)
    parts.append(llm_markdown.strip())
    parts.append("")
    parts.extend(metadata_lines)
    parts.extend(appendix_lines)
    return "\n".join(parts).rstrip() + "\n"


def _markdown_to_plain_text(text: str) -> str:
    lines_out: list[str] = []
    in_code_block = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped == "```":
            in_code_block = not in_code_block
            continue
        if stripped == "---":
            lines_out.append("")
            lines_out.append("=" * 72)
            lines_out.append("")
            continue

        line = raw
        if not in_code_block:
            hash_count = len(line) - len(line.lstrip("#"))
            if hash_count > 0:
                line = line[hash_count:].lstrip()
            if line.startswith("- "):
                line = f"* {line[2:]}"
            elif line.startswith("* "):
                line = f"* {line[2:]}"

        line = line.replace("`", "")
        lines_out.append(line)

    normalized: list[str] = []
    previous_blank = False
    for line in lines_out:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip() + "\n"


def _export_pdf_from_text(text: str, output_path: Path) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "Dependência 'reportlab' não encontrada. Instale com: pip install reportlab"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin_x = 48
    line_width_chars = 105
    y = height - 48
    c.setFont("Helvetica", 10)
    for raw_line in text.splitlines():
        logical_line = raw_line.rstrip() if raw_line else " "
        wrapped_lines = wrap(
            logical_line,
            width=line_width_chars,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [" "]
        for line in wrapped_lines:
            if y < 48:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 48
            c.drawString(margin_x, y, line)
            y -= 14
    c.save()
    return output_path


def generate_report_markdown_content(
    transcript_path: str | Path | None,
    events_csv_path: str | Path | None,
    *,
    instruments_csv_path: str | Path | None = None,
    consultation_start: datetime | None = None,
    report_profile: str | None = None,
) -> str:
    events_csv = Path(events_csv_path) if events_csv_path else None
    if events_csv is not None and not events_csv.is_file():
        events_csv = None

    transcript_p = Path(transcript_path) if transcript_path else None
    instruments_p = (
        Path(instruments_csv_path) if instruments_csv_path else None
    )
    if instruments_p is not None and not instruments_p.is_file():
        instruments_p = None

    transcript_text = (
        _load_transcript(transcript_p) if transcript_p is not None else ""
    )
    if events_csv is None and not transcript_text.strip():
        raise FileNotFoundError("Nenhum dado disponível para gerar relatório (vídeo/áudio).")
    summary = (
        aggregate_video_events(events_csv, instruments_p)
        if events_csv is not None
        else None
    )

    mode = _report_mode()
    prompt = build_prompt(
        transcript_text,
        summary,
        report_profile=report_profile,
    )
    profile = _infer_report_profile(transcript_text, summary, report_profile)

    from models.lgpd_compliance import allows_azure_openai

    azure_openai_allowed = allows_azure_openai()
    if profile == "laparo":
        llm_markdown = _build_local_fallback_llm_markdown(
            transcript=transcript_text,
            summary=summary,
            report_profile=report_profile,
        )
    elif mode == "local" or not azure_openai_allowed:
        llm_markdown = _build_local_fallback_llm_markdown(
            transcript=transcript_text,
            summary=summary,
            report_profile=report_profile,
        )
    elif mode == "hybrid":
        try:
            llm_markdown = call_llm(prompt)
        except RuntimeError:
            llm_markdown = _build_local_fallback_llm_markdown(
                transcript=transcript_text,
                summary=summary,
                report_profile=report_profile,
            )
    else:                   
        llm_markdown = call_llm(prompt)

    if not llm_markdown:
        llm_markdown = (
            "# Relatório clínico de apoio (não diagnóstico)\n\n"
            "_O modelo não devolveu conteúdo. Veja o apêndice abaixo com as "
            "métricas brutas._"
        )

    started = consultation_start or datetime.now()
    return _build_markdown_report(
        llm_markdown=llm_markdown,
        transcript=transcript_text,
        summary=summary,
        transcript_path=transcript_p,
        events_csv_path=events_csv,
        instruments_csv_path=instruments_p,
        generated_at=started,
        report_profile=report_profile,
    )


def generate_report(
    transcript_path: str | Path | None,
    events_csv_path: str | Path | None,
    *,
    instruments_csv_path: str | Path | None = None,
    report_profile: str | None = None,
    output_dir: str | Path = "data/reports",
    consultation_start: datetime | None = None,
    filename_prefix: str = "relatorio",
    output_path: str | Path | None = None,
) -> Path:
    started = consultation_start or datetime.now()
    if output_path is not None:
        out_path = Path(output_path).expanduser().resolve()
        if out_path.suffix.lower() != ".pdf":
            out_path = out_path.with_suffix(".pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = started.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{filename_prefix}_{stamp}.pdf"

    markdown = generate_report_markdown_content(
        transcript_path=transcript_path,
        events_csv_path=events_csv_path,
        instruments_csv_path=instruments_csv_path,
        consultation_start=started,
        report_profile=report_profile,
    )
    pdf_text = _markdown_to_plain_text(markdown)
    _export_pdf_from_text(pdf_text, out_path)
    return out_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fase 6: gera relatorio_<timestamp>.pdf a partir de um transcript "
            "de áudio (Fase 5) + video_events.csv (Fase 3/4)."
        )
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Caminho do transcript.txt (saída da Fase 5).",
    )
    parser.add_argument(
        "--events-csv",
        required=True,
        help="Caminho do video_events.csv (saída do pipeline de vídeo).",
    )
    parser.add_argument(
        "--instruments-csv",
        default=None,
        help=(
            "Caminho do video_instrument_events.csv (opcional, melhora a "
            "contagem por classe)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports",
        help="Diretório onde o relatório .pdf será gravado.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Caminho completo do relatório de saída (ex.: data/reports/relatorio.pdf). "
            "Se informado, sobrepõe --output-dir e --filename-prefix."
        ),
    )
    parser.add_argument(
        "--filename-prefix",
        default="relatorio",
        help='Prefixo do arquivo (default: "relatorio").',
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        out_path = generate_report(
            transcript_path=args.transcript,
            events_csv_path=args.events_csv,
            instruments_csv_path=args.instruments_csv,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
            output_path=args.output_path,
        )
    except FileNotFoundError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"Relatório gerado em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
