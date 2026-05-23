
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from models.text_summary import summarize_text

load_dotenv()


AZURE_TARGET_SAMPLE_RATE: int = 16_000
AZURE_TARGET_CHANNELS: int = 1
AZURE_TARGET_SUBTYPE: str = "PCM_16"
HESITATION_TOKENS = (
    "hmm",
    "ahn",
    "ah",
    "eh",
    "hum",
    "nao sei",
    "talvez",
    "acho que",
)
RISK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hesitacao": (
        "nao sei",
        "talvez",
        "acho que",
        "dificil de explicar",
        "nao consigo falar",
    ),
    "ansiedade": (
        "ansiosa",
        "ansiedade",
        "panico",
        "medo constante",
        "falta de ar",
        "coração acelerado",
        "coracao acelerado",
        "nervosa",
        "preocupacao excessiva",
        "preocupacao o tempo todo",
    ),
    "fadiga_extrema": (
        "exausta",
        "exausto",
        "sem energia",
        "nao consigo levantar",
        "muito cansada",
        "muito cansado",
        "insônia",
        "insonia",
        "nao durmo",
        "sono o dia inteiro",
    ),
    "abuso": (
        "agressao",
        "agrediu",
        "agredida",
        "violencia",
        "ameaça",
        "ameaca",
        "medo dele",
        "medo de voltar para casa",
        "controla tudo",
        "machucou",
        "xingamento",
    ),
    "trauma": (
        "trauma",
        "flashback",
        "pesadelo",
        "revivo isso",
        "lembranca ruim",
        "estou travada",
        "estou travado",
    ),
}
POSITIVE_TERMS = (
    "calma",
    "melhor",
    "bem",
    "tranquila",
    "tranquilo",
    "segura",
    "seguro",
)
NEGATIVE_TERMS = (
    "medo",
    "ansiedade",
    "panico",
    "triste",
    "exausta",
    "exausto",
    "dor",
    "desespero",
    "trauma",
    "violencia",
)


@dataclass
class AudioRiskAnalysis:
    sentiment_score: float
    detected_anomalies: list[str]
    confidence_level: float
    prosody: dict[str, float]
    keyword_hits: dict[str, int]
    anxiety_acoustic_score: float
    anxiety_acoustic_level: str
    risk_priority: str
    high_anxiety_detected: bool
    trauma_detected: bool
    abuse_signals_detected: bool


def _speech_config(language: str | None = None) -> speechsdk.SpeechConfig:
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    if not key or not region:
        raise RuntimeError(
            "Defina AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no arquivo .env "
            "(ou nas variáveis de ambiente do sistema)."
        )
    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    lang = (language or os.environ.get("AZURE_SPEECH_LANGUAGE", "pt-BR")).strip()
    cfg.speech_recognition_language = lang
    return cfg


def _speech_language_candidates() -> list[str]:
    primary = os.environ.get("AZURE_SPEECH_LANGUAGE", "pt-BR").strip() or "pt-BR"
    raw_fallbacks = os.environ.get("AZURE_SPEECH_FALLBACK_LANGUAGES", "en-US,es-ES")
    candidates = [primary]
    candidates.extend(item.strip() for item in raw_fallbacks.split(",") if item.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for lang in candidates:
        lowered = lang.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(lang)
    return deduped


def _transcription_quality_score(text: str) -> float:
    compact = (text or "").strip()
    if not compact:
        return float("-inf")

    tokens = re.findall(r"\b[\wÀ-ÿ']+\b", compact, flags=re.UNICODE)
    if not tokens:
        return float("-inf")

    token_count = len(tokens)
    unique_ratio = len(set(tok.lower() for tok in tokens)) / max(token_count, 1)
    alpha_chars = sum(1 for ch in compact if ch.isalpha())
    total_chars = max(len(compact), 1)
    alpha_ratio = alpha_chars / total_chars

    counts = Counter(tok.lower() for tok in tokens)
    max_freq = max(counts.values()) if counts else 0
    repetition_ratio = max_freq / max(token_count, 1)

    short_penalty = 4.0 if token_count < 3 else 0.0
    repetition_penalty = max(0.0, repetition_ratio - 0.35) * 18.0
    return (
        token_count
        + (unique_ratio * 8.0)
        + (alpha_ratio * 10.0)
        - short_penalty
        - repetition_penalty
    )


def _transcribe_once(path: Path, language: str) -> str:
    speech_config = _speech_config(language=language)
    audio_config = speechsdk.audio.AudioConfig(filename=str(path))
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    lines: list[str] = []
    done = threading.Event()
    cancel_error: list[str] = []

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                lines.append(text)

    def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
        done.set()

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        details = speechsdk.CancellationDetails.from_result(evt.result)
        if details.reason == speechsdk.CancellationReason.Error:
            cancel_error.append(
                f"Erro do Azure Speech: {details.error_details} "
                "(verifique chave, região e créditos da conta)."
            )
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition()
    while not done.wait(timeout=0.25):
        pass
    recognizer.stop_continuous_recognition()
    time.sleep(0.2)

    if cancel_error:
        raise RuntimeError(cancel_error[0])
    return " ".join(lines).strip()


def _ensure_azure_wav(audio_path: Path) -> Path:
    try:
        import soundfile as sf                
    except ImportError as exc:                                            
        raise RuntimeError(
            "Pacote 'soundfile' ausente. Instale com `pip install soundfile`."
        ) from exc

    try:
        info = sf.info(str(audio_path))
    except Exception:
        info = None

    is_already_ok = (
        info is not None
        and audio_path.suffix.lower() == ".wav"
        and info.samplerate == AZURE_TARGET_SAMPLE_RATE
        and info.channels == AZURE_TARGET_CHANNELS
        and info.subtype == AZURE_TARGET_SUBTYPE
    )
    if is_already_ok:
        return audio_path

    try:
        import librosa                
    except ImportError as exc:                                            
        raise RuntimeError(
            "Pacote 'librosa' ausente. Instale com `pip install librosa`."
        ) from exc

    samples, _ = librosa.load(
        str(audio_path),
        sr=AZURE_TARGET_SAMPLE_RATE,
        mono=True,
    )

    target = audio_path.with_name(f"{audio_path.stem}_16k_mono.wav")
    sf.write(
        str(target),
        samples,
        AZURE_TARGET_SAMPLE_RATE,
        subtype=AZURE_TARGET_SUBTYPE,
    )
    return target


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", normalized).strip()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    return float(numerator) / float(denominator)


def _extract_prosodic_features(audio_path: Path) -> dict[str, float]:
    try:
        import librosa                
        import numpy as np                
    except ImportError:
        return {
            "duration_sec": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "jitter_local_pct": 0.0,
            "shimmer_local_pct": 0.0,
            "speech_rate_approx_wps": 0.0,
            "silence_ratio": 0.0,
        }

    signal, sr = librosa.load(str(audio_path), sr=AZURE_TARGET_SAMPLE_RATE, mono=True)
    duration = float(librosa.get_duration(y=signal, sr=sr) or 0.0)
    if duration <= 1e-6 or signal.size == 0:
        return {
            "duration_sec": duration,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "jitter_local_pct": 0.0,
            "shimmer_local_pct": 0.0,
            "speech_rate_approx_wps": 0.0,
            "silence_ratio": 0.0,
        }

    hop = 256
    frame_length = 1024
    pitch_track = librosa.yin(
        signal,
        fmin=80,
        fmax=350,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop,
    )
    finite_pitch = pitch_track[np.isfinite(pitch_track)]
    pitch_mean = float(np.mean(finite_pitch)) if finite_pitch.size else 0.0
    pitch_std = float(np.std(finite_pitch)) if finite_pitch.size else 0.0

    periods = 1.0 / np.clip(finite_pitch, 1e-6, None) if finite_pitch.size else np.array([])
    jitter = (
        float(np.mean(np.abs(np.diff(periods)) / np.clip(periods[:-1], 1e-6, None)) * 100.0)
        if periods.size > 1
        else 0.0
    )

    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop).flatten()
    shimmer = (
        float(np.mean(np.abs(np.diff(rms)) / np.clip(rms[:-1], 1e-6, None)) * 100.0)
        if rms.size > 1
        else 0.0
    )
    silence_threshold = max(float(np.percentile(rms, 20) * 0.8), 1e-6) if rms.size else 1e-6
    silence_ratio = float(np.mean(rms < silence_threshold)) if rms.size else 0.0

    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    speech_rate = float(len(onset_frames) / max(duration, 1e-6))

    return {
        "duration_sec": round(duration, 3),
        "pitch_mean_hz": round(pitch_mean, 3),
        "pitch_std_hz": round(pitch_std, 3),
        "jitter_local_pct": round(jitter, 3),
        "shimmer_local_pct": round(shimmer, 3),
        "speech_rate_approx_wps": round(speech_rate, 3),
        "silence_ratio": round(silence_ratio, 3),
    }


def _estimate_sentiment_score(text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer                

        vader = SentimentIntensityAnalyzer()
        return round(float(vader.polarity_scores(normalized)["compound"]), 4)
    except Exception:
        pass

    positive_hits = sum(normalized.count(term) for term in POSITIVE_TERMS)
    negative_hits = sum(normalized.count(term) for term in NEGATIVE_TERMS)
    score = _safe_ratio(positive_hits - negative_hits, positive_hits + negative_hits + 1)
    return round(max(-1.0, min(1.0, score)), 4)


def _keyword_risk_scan(text: str) -> dict[str, int]:
    normalized = _normalize_text(text)
    counts: dict[str, int] = {}
    for category, keywords in RISK_KEYWORDS.items():
        counts[category] = sum(normalized.count(keyword) for keyword in keywords)

    hesitation_noise = sum(normalized.count(token) for token in HESITATION_TOKENS)
    counts["hesitacao"] += hesitation_noise
    counts["hesitacao"] += len(re.findall(r"\.\.\.", text))
    return counts


def _detected_anomalies_from_signals(
    keyword_hits: dict[str, int],
    prosody: dict[str, float],
    anxiety_acoustic_score: float,
    transcript: str,
) -> list[str]:
    anomalies: list[str] = []
    if keyword_hits.get("hesitacao", 0) >= 2 or prosody.get("silence_ratio", 0.0) > 0.45:
        anomalies.append("hesitação detectada")
    if keyword_hits.get("ansiedade", 0) >= 1 or anxiety_acoustic_score >= 0.55:
        anomalies.append("sinais de ansiedade")
    if keyword_hits.get("fadiga_extrema", 0) >= 1 or prosody.get("speech_rate_approx_wps", 0.0) < 1.2:
        anomalies.append("fadiga extrema provável")
    if keyword_hits.get("abuso", 0) >= 1:
        anomalies.append("sinal de abuso verbalizado")
    if keyword_hits.get("trauma", 0) >= 1:
        anomalies.append("trauma psicológico indicado")
    if not transcript.strip():
        anomalies.append("sem fala válida para análise")
    return anomalies


def _derive_risk_priority(
    sentiment_score: float,
    keyword_hits: dict[str, int],
    anomalies: list[str],
) -> str:
    if keyword_hits.get("abuso", 0) >= 1 or keyword_hits.get("trauma", 0) >= 1:
        return "critical"
    if keyword_hits.get("ansiedade", 0) >= 2:
        return "high"
    if sentiment_score <= -0.45 and len(anomalies) >= 2:
        return "high"
    if len(anomalies) >= 2:
        return "moderate"
    return "low"


def _estimate_confidence_level(keyword_hits: dict[str, int], prosody: dict[str, float]) -> float:
    signals = (
        min(keyword_hits.get("hesitacao", 0), 3)
        + min(keyword_hits.get("ansiedade", 0), 3)
        + min(keyword_hits.get("fadiga_extrema", 0), 3)
        + min(keyword_hits.get("abuso", 0), 2) * 2
        + min(keyword_hits.get("trauma", 0), 2) * 2
    )
    prosody_bonus = 0.0
    if prosody.get("jitter_local_pct", 0.0) > 2.0:
        prosody_bonus += 0.15
    if prosody.get("silence_ratio", 0.0) > 0.4:
        prosody_bonus += 0.15
    if prosody.get("speech_rate_approx_wps", 0.0) < 1.1:
        prosody_bonus += 0.1
    base = min(0.95, 0.25 + (signals / 12.0) + prosody_bonus)
    return round(max(0.05, base), 3)


def _estimate_anxiety_acoustic_score(prosody: dict[str, float]) -> tuple[float, str]:
    jitter = float(prosody.get("jitter_local_pct", 0.0) or 0.0)
    shimmer = float(prosody.get("shimmer_local_pct", 0.0) or 0.0)
    pitch_std = float(prosody.get("pitch_std_hz", 0.0) or 0.0)
    speech_rate = float(prosody.get("speech_rate_approx_wps", 0.0) or 0.0)

    score = 0.0

                                                                 
    if jitter >= 3.0:
        score += 0.35
    elif jitter >= 2.0:
        score += 0.2

                                                  
    if shimmer >= 8.0:
        score += 0.25
    elif shimmer >= 5.0:
        score += 0.12

                                                                     
    if pitch_std >= 65.0:
        score += 0.22
    elif pitch_std >= 45.0:
        score += 0.1

                                                                   
    if speech_rate >= 4.2:
        score += 0.18
    elif speech_rate >= 3.6:
        score += 0.08

    normalized = round(max(0.0, min(1.0, score)), 3)
    if normalized >= 0.7:
        level = "high"
    elif normalized >= 0.45:
        level = "moderate"
    else:
        level = "low"
    return normalized, level


def analyze_audio_risk(audio_path: str | Path, transcript_text: str) -> AudioRiskAnalysis:
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado para análise acústica: {path}")

    prosody = _extract_prosodic_features(path)
    keyword_hits = _keyword_risk_scan(transcript_text)
    sentiment_score = _estimate_sentiment_score(transcript_text)
    anxiety_acoustic_score, anxiety_acoustic_level = _estimate_anxiety_acoustic_score(prosody)
    anomalies = _detected_anomalies_from_signals(
        keyword_hits,
        prosody,
        anxiety_acoustic_score,
        transcript_text,
    )
    confidence = _estimate_confidence_level(keyword_hits, prosody)
    priority = _derive_risk_priority(sentiment_score, keyword_hits, anomalies)
    high_anxiety = keyword_hits.get("ansiedade", 0) >= 2 or anxiety_acoustic_score >= 0.7

    return AudioRiskAnalysis(
        sentiment_score=sentiment_score,
        detected_anomalies=anomalies,
        confidence_level=confidence,
        prosody=prosody,
        keyword_hits=keyword_hits,
        anxiety_acoustic_score=anxiety_acoustic_score,
        anxiety_acoustic_level=anxiety_acoustic_level,
        risk_priority=priority,
        high_anxiety_detected=high_anxiety,
        trauma_detected=keyword_hits.get("trauma", 0) >= 1,
        abuse_signals_detected=keyword_hits.get("abuso", 0) >= 1,
    )


def save_audio_analysis_with_timestamp(
    analysis: AudioRiskAnalysis,
    output_dir: str | Path,
    *,
    consultation_start: datetime | None = None,
    prefix: str = "audio_analysis",
) -> Path:
    started = consultation_start or datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{stamp}.json"
    payload = asdict(analysis)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def transcribe_file(
    audio_path: str | Path,
    *,
    ensure_wav_16k_mono: bool = True,
) -> str:
    from models.lgpd_compliance import assert_azure_speech_allowed

    assert_azure_speech_allowed()

    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    if ensure_wav_16k_mono:
        path = _ensure_azure_wav(path)

    best_text = ""
    best_score = float("-inf")
    errors: list[str] = []

    for language in _speech_language_candidates():
        try:
            candidate = _transcribe_once(path, language)
        except Exception as exc:
            errors.append(f"[{language}] {exc}")
            continue

        score = _transcription_quality_score(candidate)
        if score > best_score:
            best_text = candidate
            best_score = score

    if best_text:
        return best_text
    if errors:
        raise RuntimeError(errors[0])
    return ""


def save_transcript_with_timestamp(
    text: str,
    output_dir: str | Path,
    *,
    consultation_start: datetime | None = None,
    prefix: str = "transcript",
) -> Path:
    started = consultation_start or datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{prefix}_{stamp}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fase 5: transcreve um áudio com Azure Speech e (opcional) salva "
            "transcript_<timestamp>.txt em disco."
        )
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Caminho do arquivo de áudio (ex.: data/audios/audio_teste_speech.wav)",
    )
    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help=(
            "Salva a transcrição em disco como transcript_<YYYYmmdd_HHMMSS>.txt. "
            "Usa o timestamp do início da consulta no nome do arquivo."
        ),
    )
    parser.add_argument(
        "--transcript-dir",
        default="data/transcripts",
        help=(
            "Diretório onde o arquivo transcript_<timestamp>.txt será gravado. "
            "Default: data/transcripts."
        ),
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help=(
            "Não converter o áudio para WAV 16 kHz mono antes de enviar ao "
            "Azure. Use apenas se o arquivo já estiver nesse formato."
        ),
    )
    parser.add_argument(
        "--resumir",
        action="store_true",
        help="Gera um resumo via Azure OpenAI após a transcrição ficar pronta.",
    )
    parser.add_argument(
        "--save-analysis",
        action="store_true",
        help=(
            "Executa análise prosódica + risco textual e salva "
            "audio_analysis_<timestamp>.json."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        default="data/audio_analysis",
        help="Diretório para salvar o JSON de análise de áudio.",
    )
    parser.add_argument(
        "--resumo-idioma",
        choices=("pt", "en"),
        default="pt",
        help="Idioma do resumo (padrão: pt).",
    )
    args = parser.parse_args(argv)

    if not args.audio:
        print(
            "Informe o caminho do áudio.\n"
            "Exemplo: python -m models.audio_analysis data/audios/audio_teste_speech.wav",
            file=sys.stderr,
        )
        return 1

    consultation_start = datetime.now()

    try:
        texto = transcribe_file(
            args.audio,
            ensure_wav_16k_mono=not args.no_convert,
        )
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if not texto:
        print(
            "(Nenhuma fala reconhecida — áudio muito baixo, silêncio ou idioma incorreto.)"
        )
        return 0

    print(texto)

    if args.save_transcript:
        try:
            out_path = save_transcript_with_timestamp(
                texto,
                args.transcript_dir,
                consultation_start=consultation_start,
            )
        except OSError as exc:
            print(f"Falha ao gravar transcript: {exc}", file=sys.stderr)
            return 1
        print(f"\nTranscript salvo em: {out_path}")

    if args.save_analysis:
        try:
            analysis = analyze_audio_risk(args.audio, texto)
            analysis_path = save_audio_analysis_with_timestamp(
                analysis,
                args.analysis_dir,
                consultation_start=consultation_start,
            )
        except Exception as exc:
            print(f"Falha na análise de áudio: {exc}", file=sys.stderr)
            return 1
        print("\n--- ANALISE DE RISCO DE AUDIO ---")
        print(json.dumps(asdict(analysis), ensure_ascii=False, indent=2))
        print(f"JSON salvo em: {analysis_path}")

    if args.resumir:
        try:
            resumo = summarize_text(texto, language=args.resumo_idioma)
        except Exception as exc:
            print(f"Falha ao resumir com Azure OpenAI: {exc}", file=sys.stderr)
            return 1
        if resumo:
            print("\n--- RESUMO ---")
            print(resumo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
