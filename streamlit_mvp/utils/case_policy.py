from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORY_ACOLHIMENTO = "Acolhimento/Violência"
CATEGORY_DOR = "Dor corporal"
CATEGORY_LAPARO = "Laparoscopia ginecológica"
CATEGORY_INDETERMINADO = "indeterminado"

CASE_CATEGORIES: tuple[str, ...] = (
    CATEGORY_ACOLHIMENTO,
    CATEGORY_DOR,
    CATEGORY_LAPARO,
)

MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_LAPARO = 0.45


@dataclass(frozen=True)
class PainBehaviorVisualThresholds:
    """Limiares visuais menos sensíveis para acolhimento/dor (vs. laparoscopia)."""

    blood_score: float
    blood_persist: float
    bruise_score: float
    bruise_min_pixels: int
    alert_min_blood_frames: int
    alert_min_bruise_frames: int
    alert_min_bruise_score: float
    summary_min_blood_frames: int
    summary_min_blood_score: float
    summary_min_bruise_frames: int
    summary_min_bruise_score: float


PAIN_BEHAVIOR_THRESHOLDS = PainBehaviorVisualThresholds(
    blood_score=0.92,
    blood_persist=0.95,
    bruise_score=0.18,
    bruise_min_pixels=28,
    alert_min_blood_frames=5,
    alert_min_bruise_frames=10,
    alert_min_bruise_score=0.18,
    summary_min_blood_frames=5,
    summary_min_blood_score=0.92,
    summary_min_bruise_frames=8,
    summary_min_bruise_score=0.18,
)

# Dor corporal: hematoma equilibrado (menos FP que acolhimento, mais recall que laparo).
DOR_CORPORAL_THRESHOLDS = PainBehaviorVisualThresholds(
    blood_score=0.92,
    blood_persist=0.95,
    bruise_score=0.22,
    bruise_min_pixels=30,
    alert_min_blood_frames=5,
    alert_min_bruise_frames=11,
    alert_min_bruise_score=0.22,
    summary_min_blood_frames=5,
    summary_min_blood_score=0.92,
    summary_min_bruise_frames=9,
    summary_min_bruise_score=0.22,
)


def get_pain_behavior_thresholds(category: str | None) -> PainBehaviorVisualThresholds:
    normalized = normalize_case_category(category)
    if normalized == CATEGORY_DOR:
        return DOR_CORPORAL_THRESHOLDS
    return PAIN_BEHAVIOR_THRESHOLDS


@dataclass(frozen=True)
class CategoryPolicy:
    allow_instrument_identification: bool
    allow_pain_and_pose: bool
    allow_behavioral_signals: bool
    allow_audio_transcript: bool
    allow_bruise_detection: bool = True


def normalize_case_category(raw_value: str | None) -> str:
    value = (raw_value or "").strip().lower()
    if "acolhimento" in value or "viol" in value:
        return CATEGORY_ACOLHIMENTO
    if "dor" in value:
        return CATEGORY_DOR
    if "laparo" in value or "gineco" in value or "cirurg" in value:
        return CATEGORY_LAPARO
    return CATEGORY_INDETERMINADO


def get_case_policy(category: str | None) -> CategoryPolicy:
    normalized = normalize_case_category(category)
    if normalized == CATEGORY_ACOLHIMENTO:
        return CategoryPolicy(
            allow_instrument_identification=False,
            allow_pain_and_pose=True,
            allow_behavioral_signals=True,
            allow_audio_transcript=True,
        )
    if normalized == CATEGORY_DOR:
        return CategoryPolicy(
            allow_instrument_identification=False,
            allow_pain_and_pose=True,
            allow_behavioral_signals=True,
            allow_audio_transcript=True,
        )
    if normalized == CATEGORY_LAPARO:
        return CategoryPolicy(
            allow_instrument_identification=True,
            allow_pain_and_pose=False,
            allow_behavioral_signals=False,
            allow_audio_transcript=False,
        )
    return CategoryPolicy(
        allow_instrument_identification=False,
        allow_pain_and_pose=False,
        allow_behavioral_signals=False,
        allow_audio_transcript=False,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _row_blood_flag(row: dict[str, Any]) -> int:
    return _safe_int(
        row.get("flag_sangue_identificado", row.get("visual_blood_detected", 0)),
        0,
    )


def _row_blood_score(row: dict[str, Any]) -> float:
    return _safe_float(
        row.get("score_sangue_identificado", row.get("visual_blood_score", 0.0)),
        0.0,
    )


def _row_bruise_flag(row: dict[str, Any]) -> int:
    return _safe_int(
        row.get("flag_equimose_identificada", row.get("flag_hematoma_identificado", 0)),
        0,
    )


def _row_bruise_score(row: dict[str, Any]) -> float:
    return _safe_float(
        row.get(
            "score_equimose_identificada",
            row.get("score_hematoma_identificado", 0.0),
        ),
        0.0,
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _build_vision_payload(events_rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotion_columns = (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
    )

    def _empty_emotion_aggregate() -> dict[str, dict[str, float]]:
        return {
            emotion: {"mean": 0.0, "max": 0.0, "last": 0.0}
            for emotion in emotion_columns
        }

    if not events_rows:
        return {
            "available": False,
            "frames_analyzed": 0,
            "emotion": {
                "fear_probability_max": 0.0,
                "fear_probability_last": 0.0,
                "face_detected_frames": 0,
            },
            "pose": {
                "curvature_max_deg": 0.0,
                "curvature_last_deg": 0.0,
            },
            "decision": {
                "critical_alert_frames": 0,
                "critical_alert_last": False,
            },
            "emotion_probabilities": _empty_emotion_aggregate(),
        }

    fear_values = [_safe_float(row.get("emotion_fear_prob"), 0.0) for row in events_rows]
    curvature_values = [_safe_float(row.get("pose_curvature_deg"), 0.0) for row in events_rows]
    face_detected_frames = sum(_safe_int(row.get("emotion_face_detected"), 0) for row in events_rows)
    critical_alert_frames = sum(_safe_int(row.get("vision_critical_alert"), 0) for row in events_rows)
    last = events_rows[-1]
    emotion_probabilities: dict[str, dict[str, float]] = {}
    for emotion in emotion_columns:
        col = f"emotion_{emotion}_prob"
        values = [_safe_float(row.get(col), 0.0) for row in events_rows]
        mean_value = (sum(values) / len(values)) if values else 0.0
        emotion_probabilities[emotion] = {
            "mean": round(mean_value, 4),
            "max": round(max(values, default=0.0), 4),
            "last": round(_safe_float(last.get(col), 0.0), 4),
        }

    return {
        "available": True,
        "frames_analyzed": len(events_rows),
        "last_frame": {
            "frame_idx": _safe_int(last.get("frame_idx"), -1),
            "timestamp_sec": round(_safe_float(last.get("timestamp_sec"), 0.0), 3),
            "fear_probability": round(_safe_float(last.get("emotion_fear_prob"), 0.0), 4),
            "face_detected": bool(_safe_int(last.get("emotion_face_detected"), 0) == 1),
            "curvature_deg": round(_safe_float(last.get("pose_curvature_deg"), 0.0), 3),
            "critical_alert": bool(_safe_int(last.get("vision_critical_alert"), 0) == 1),
        },
        "emotion": {
            "fear_probability_max": round(max(fear_values, default=0.0), 4),
            "fear_probability_last": round(_safe_float(last.get("emotion_fear_prob"), 0.0), 4),
            "face_detected_frames": int(face_detected_frames),
        },
        "pose": {
            "curvature_max_deg": round(max(curvature_values, default=0.0), 3),
            "curvature_last_deg": round(_safe_float(last.get("pose_curvature_deg"), 0.0), 3),
        },
        "decision": {
            "critical_alert_frames": int(critical_alert_frames),
            "critical_alert_last": bool(_safe_int(last.get("vision_critical_alert"), 0) == 1),
        },
        "emotion_probabilities": emotion_probabilities,
    }


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


EVENTS_CSV_COLUMNS_EN_TO_PT: dict[str, str] = {
    "pain_proxy": "proxy_dor",
    "pain_smoothed": "dor_suavizada_pct",
    "behavior_score": "score_postural",
    "flag_pose_visible": "flag_pose_visivel",
    "flag_left_arm_bent": "flag_braco_esquerdo_dobrado",
    "flag_right_arm_bent": "flag_braco_direito_dobrado",
    "flag_curved_posture": "flag_postura_curvada",
    "flag_asymmetric_shoulders": "flag_assimetria_ombros",
    "n_faces": "qtd_rostos",
    "visual_aggression_score": "score_agressao_visual",
    "visual_aggression_persisted": "score_agressao_visual_persistido",
    "score_sangue_identificado": "score_sangue_identificado",
    "flag_sangue_identificado": "flag_sangue_identificado",
    "visual_blood_score": "score_sangue_visual",
    "visual_blood_detected": "flag_sangue_visual",
    "score_equimose_identificada": "score_equimose_identificada",
    "flag_equimose_identificada": "flag_equimose_identificada",
    "emotion_angry_prob": "prob_emocao_raiva",
    "emotion_disgust_prob": "prob_emocao_nojo",
    "emotion_fear_prob": "prob_emocao_medo",
    "emotion_happy_prob": "prob_emocao_felicidade",
    "emotion_neutral_prob": "prob_emocao_neutro",
    "emotion_sad_prob": "prob_emocao_tristeza",
    "emotion_face_detected": "flag_rosto_detectado",
    "pose_curvature_deg": "curvatura_postural_graus",
    "vision_critical_alert": "alerta_critico_visao",
}

EVENTS_CSV_COLUMNS_PT_TO_EN: dict[str, str] = {
    pt_name: en_name for en_name, pt_name in EVENTS_CSV_COLUMNS_EN_TO_PT.items()
}

LAPAROSCOPY_DISABLED_COLUMNS_PT: dict[str, Any] = {
    "proxy_dor": "",
    "dor_suavizada_pct": 0.0,
    "score_postural": 0.0,
    "flag_pose_visivel": 0,
    "flag_braco_esquerdo_dobrado": 0,
    "flag_braco_direito_dobrado": 0,
    "flag_postura_curvada": 0,
    "flag_assimetria_ombros": 0,
    "qtd_rostos": 0,
    "score_agressao_visual": 0.0,
    "score_agressao_visual_persistido": 0.0,
    "score_sangue_identificado": 0.0,
    "flag_sangue_identificado": 0,
    "score_sangue_visual": 0.0,
    "flag_sangue_visual": 0,
    "score_equimose_identificada": 0.0,
    "flag_equimose_identificada": 0,
    "prob_emocao_raiva": 0.0,
    "prob_emocao_nojo": 0.0,
    "prob_emocao_medo": 0.0,
    "prob_emocao_felicidade": 0.0,
    "prob_emocao_neutro": 0.0,
    "prob_emocao_tristeza": 0.0,
    "flag_rosto_detectado": 0,
    "curvatura_postural_graus": 0.0,
    "alerta_critico_visao": 0,
}


def rename_events_csv_columns_to_en(fieldnames: list[str]) -> list[str]:
    return [EVENTS_CSV_COLUMNS_PT_TO_EN.get(name, name) for name in fieldnames]


def rename_events_csv_row_to_en(row: dict[str, Any]) -> dict[str, Any]:
    return {
        EVENTS_CSV_COLUMNS_PT_TO_EN.get(key, key): value
        for key, value in row.items()
    }


_AZURE_ACOES_CATEGORIES = frozenset({CATEGORY_ACOLHIMENTO, CATEGORY_DOR})
_ACOES_LLM_SYSTEM_PROMPT = (
    "Você apoia triagem clínica não diagnóstica em saúde da mulher. "
    "Com base somente nos dados estruturados fornecidos, produza de 3 a 5 "
    "ações prioritárias objetivas para a equipe humana. "
    "Não diagnostique, não prescreva medicamento e não substitua avaliação "
    "profissional presencial."
)


def _risk_alert_llm_enabled() -> bool:
    from models.lgpd_compliance import allows_azure_openai

    mode = os.environ.get("REPORT_MODE", "local").strip().lower()
    return mode in {"azure", "hybrid"} and allows_azure_openai()


def _parse_acoes_prioritarias_from_llm(
    text: str,
    fallback: list[str],
) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[-*•]\s+", "", stripped)
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
        if stripped:
            lines.append(stripped)
    if len(lines) >= 2:
        return lines[:5]
    return fallback


def _build_acoes_prioritarias_prompt(
    output: dict[str, Any],
    audio_payload: dict[str, Any],
) -> str:
    from models.lgpd_compliance import sanitize_external_json_payload

    audio_summary = sanitize_external_json_payload(
        {
            "sentiment_score": audio_payload.get("sentiment_score"),
            "confidence_level": audio_payload.get("confidence_level"),
            "risk_priority": audio_payload.get("risk_priority"),
            "detected_anomalies": audio_payload.get("detected_anomalies", []),
            "high_anxiety_detected": audio_payload.get("high_anxiety_detected"),
            "trauma_detected": audio_payload.get("trauma_detected"),
            "abuse_signals_detected": audio_payload.get("abuse_signals_detected"),
        }
    )
    safe_output = sanitize_external_json_payload(
        {
            "categoria": output.get("categoria"),
            "priority": output.get("priority"),
            "alertas": output.get("alertas"),
            "resumo_tecnico": output.get("resumo_tecnico"),
            "vision": output.get("vision", {}),
            "acoes_locais_referencia": output.get("acoes_prioritarias", []),
        }
    )
    return (
        "Gere apenas uma lista numerada de ações prioritárias (3 a 5 itens), "
        "sem introdução e sem conclusão.\n\n"
        f"Dados estruturados (JSON): {json.dumps(safe_output, ensure_ascii=False)}\n"
        f"Sinais de áudio agregados (JSON): {json.dumps(audio_summary, ensure_ascii=False)}"
    )


def _enrich_case_output_with_azure(
    output: dict[str, Any],
    *,
    audio_payload: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(output)
    fallback = enriched.get("acoes_prioritarias", [])
    if not isinstance(fallback, list):
        fallback = [str(fallback)] if fallback else []

    categoria = str(enriched.get("categoria", ""))
    if (
        not _risk_alert_llm_enabled()
        or categoria not in _AZURE_ACOES_CATEGORIES
        or not fallback
    ):
        enriched["acoes_prioritarias_fonte"] = "local"
        return enriched

    try:
        from models.report_generator import call_llm

        prompt = _build_acoes_prioritarias_prompt(enriched, audio_payload)
        response = call_llm(
            prompt,
            temperature=0.15,
            max_tokens=420,
            system_prompt=_ACOES_LLM_SYSTEM_PROMPT,
        )
        parsed = _parse_acoes_prioritarias_from_llm(response, fallback)
        if parsed != fallback:
            enriched["acoes_prioritarias"] = parsed
            enriched["acoes_prioritarias_fonte"] = "azure"
        else:
            enriched["acoes_prioritarias_fonte"] = "local"
    except Exception:
        enriched["acoes_prioritarias"] = fallback
        enriched["acoes_prioritarias_fonte"] = "local"

    return enriched


def build_structured_case_output(
    *,
    category: str | None,
    events_csv: Path,
    instrument_csv: Path,
    audio_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    from models.lgpd_compliance import build_lgpd_audit_metadata

    payload = _build_structured_case_output_core(
        category=category,
        events_csv=events_csv,
        instrument_csv=instrument_csv,
        audio_analysis=audio_analysis,
    )
    payload = _enrich_case_output_with_azure(
        payload,
        audio_payload=_as_mapping(audio_analysis),
    )
    payload["lgpd"] = build_lgpd_audit_metadata()
    return payload


def _rename_events_csv_columns_to_pt(
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    renamed_fieldnames: list[str] = []
    for name in fieldnames:
        pt_name = EVENTS_CSV_COLUMNS_EN_TO_PT.get(name, name)
        if pt_name not in renamed_fieldnames:
            renamed_fieldnames.append(pt_name)

    renamed_rows: list[dict[str, Any]] = []
    for row in rows:
        renamed_row: dict[str, Any] = {}
        for key, value in row.items():
            pt_key = EVENTS_CSV_COLUMNS_EN_TO_PT.get(key, key)
            renamed_row[pt_key] = value
        for pt_col, default in LAPAROSCOPY_DISABLED_COLUMNS_PT.items():
            renamed_row[pt_col] = default
            if pt_col not in renamed_fieldnames:
                renamed_fieldnames.append(pt_col)
        renamed_rows.append(renamed_row)

    return renamed_fieldnames, renamed_rows


def _sanitize_events_for_non_instrument_modes(events_csv: Path) -> None:
    rows = _read_csv_rows(events_csv)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        row["n_instruments"] = 0
        row["instruments_seen"] = ""
        row["instrument_top_conf"] = 0.0
    _write_csv_rows(events_csv, fieldnames, rows)


def _sanitize_events_for_laparoscopy(events_csv: Path) -> None:
    rows = _read_csv_rows(events_csv)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    fieldnames, rows = _rename_events_csv_columns_to_pt(fieldnames, rows)
    _write_csv_rows(events_csv, fieldnames, rows)


def _clear_instrument_csv(instrument_csv: Path) -> None:
    if not instrument_csv.exists():
        return
    rows = _read_csv_rows(instrument_csv)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "frame_idx",
            "timestamp_sec",
            "timestamp",
            "class_id",
            "class_name",
            "class_group",
            "confidence",
            "spectral_confirmed",
            "rg_ratio",
            "rb_ratio",
            "red_mean",
            "pixel_count",
            "risk_alert",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    _write_csv_rows(instrument_csv, fieldnames, [])


def _mark_unmapped_instruments(instrument_csv: Path) -> None:
    rows = _read_csv_rows(instrument_csv)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        group = str(row.get("class_group", "")).strip().lower()
        confidence = _safe_float(row.get("confidence"), 0.0)
        if group not in {"instrumento", "bleeding"} and confidence >= MIN_CONFIDENCE:
            row["class_group"] = "instrumento"
            row["class_name"] = "instrumento cirúrgico não identificado"
    _write_csv_rows(instrument_csv, fieldnames, rows)


def enforce_case_policy_outputs(
    *,
    category: str | None,
    events_csv: Path,
    instrument_csv: Path,
) -> None:
    normalized = normalize_case_category(category)
    if normalized in {CATEGORY_ACOLHIMENTO, CATEGORY_DOR}:
        _sanitize_events_for_non_instrument_modes(events_csv)
        _clear_instrument_csv(instrument_csv)
        return
    if normalized == CATEGORY_LAPARO:
        _sanitize_events_for_laparoscopy(events_csv)
        _mark_unmapped_instruments(instrument_csv)


def _build_structured_case_output_core(
    *,
    category: str | None,
    events_csv: Path,
    instrument_csv: Path,
    audio_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_case_category(category)
    events_rows = [
        rename_events_csv_row_to_en(row) for row in _read_csv_rows(events_csv)
    ]
    instrument_rows = _read_csv_rows(instrument_csv)
    audio_payload = _as_mapping(audio_analysis)
    vision_payload = _build_vision_payload(events_rows)

    audio_confidence = _safe_float(audio_payload.get("confidence_level"), 0.0)
    sentiment = _safe_float(audio_payload.get("sentiment_score"), 0.0)
    anomalies = audio_payload.get("detected_anomalies", [])
    if not isinstance(anomalies, list):
        anomalies = [str(anomalies)] if anomalies else []

    if normalized == CATEGORY_ACOLHIMENTO:
        max_aggression = max(
            (_safe_float(row.get("visual_aggression_persisted"), 0.0) for row in events_rows),
            default=0.0,
        )
        posture_flags = sum(
            _safe_int(row.get("flag_curved_posture"), 0) + _safe_int(row.get("flag_asymmetric_shoulders"), 0)
            for row in events_rows
        )
        blood_frames = sum(_row_blood_flag(row) for row in events_rows)
        bruise_frames = sum(_row_bruise_flag(row) for row in events_rows)
        max_bruise_score = max((_row_bruise_score(row) for row in events_rows), default=0.0)
        confidence = max(max_aggression, audio_confidence)
        if confidence < MIN_CONFIDENCE:
            vision_payload = dict(vision_payload)
            vision_payload.pop("emotion_probabilities", None)
            return {
                "categoria": CATEGORY_INDETERMINADO,
                "acoes_prioritarias": [
                    "Solicitar revisão humana com amostra adicional de vídeo/áudio.",
                    "Reexecutar inferência com melhor qualidade de captura.",
                ],
                "alertas": False,
                "resumo_tecnico": "Baixa confiança para padrões emocionais/comportamentais (>70% não atingido).",
                "priority": "low",
                "vision": vision_payload,
            }
        alertas = bool(
            max_aggression >= 0.65
            or blood_frames >= PAIN_BEHAVIOR_THRESHOLDS.alert_min_blood_frames
            or bruise_frames >= PAIN_BEHAVIOR_THRESHOLDS.alert_min_bruise_frames
            or max_bruise_score >= PAIN_BEHAVIOR_THRESHOLDS.alert_min_bruise_score
            or audio_payload.get("abuse_signals_detected", False)
            or audio_payload.get("trauma_detected", False)
        )
        vision_payload = dict(vision_payload)
        vision_payload.pop("emotion_probabilities", None)
        return {
            "categoria": CATEGORY_ACOLHIMENTO,
            "acoes_prioritarias": [
                "Priorizar acolhimento com escuta ativa e ambiente seguro.",
                "Encaminhar avaliação psicossocial quando houver sinais de trauma/abuso.",
                "Registrar evolução comportamental para comparação temporal.",
            ],
            "alertas": alertas,
            "resumo_tecnico": (
                "Análise focada em comportamento/emoção: "
                f"agressão_visual_max={max_aggression:.3f}, "
                f"flags_posturais={posture_flags}, "
                f"frames_sangue_facial={blood_frames}, "
                f"frames_hematoma_facial={bruise_frames}, "
                f"score_hematoma_max={max_bruise_score:.3f}, "
                f"conf_audio={audio_confidence:.3f}."
            ),
            "priority": "high" if alertas else "moderate",
            "vision": vision_payload,
        }

    if normalized == CATEGORY_DOR:
        dor_visual = get_pain_behavior_thresholds(normalized)
        pain_max = max((_safe_float(row.get("pain_smoothed"), 0.0) for row in events_rows), default=0.0)
        suffering_frames = sum(1 for row in events_rows if _safe_float(row.get("pain_smoothed"), 0.0) >= 40.0)
        posture_changes = sum(
            1
            for row in events_rows
            if _safe_int(row.get("flag_curved_posture"), 0) == 1 or _safe_int(row.get("flag_asymmetric_shoulders"), 0) == 1
        )
        blood_frames = sum(_row_blood_flag(row) for row in events_rows)
        bruise_frames = sum(_row_bruise_flag(row) for row in events_rows)
        max_bruise_score = max((_row_bruise_score(row) for row in events_rows), default=0.0)
        confidence = max(min(pain_max / 100.0, 1.0), audio_confidence)
        if confidence < MIN_CONFIDENCE:
            return {
                "categoria": CATEGORY_INDETERMINADO,
                "acoes_prioritarias": [
                    "Solicitar nova amostra multimodal para dor corporal.",
                    "Revisar enquadramento de pose e captação de áudio.",
                ],
                "alertas": False,
                "resumo_tecnico": "Sinais de dor corporal sem confiança mínima de 70%.",
                "priority": "low",
                "vision": vision_payload,
            }
        alertas = bool(
            pain_max >= 40.0
            or len(anomalies) >= 2
            or sentiment <= -0.45
            or blood_frames >= dor_visual.alert_min_blood_frames
            or bruise_frames >= dor_visual.alert_min_bruise_frames
            or max_bruise_score >= dor_visual.alert_min_bruise_score
        )
        return {
            "categoria": CATEGORY_DOR,
            "acoes_prioritarias": [
                "Sugerir posição de conforto com apoio lombar e respiração guiada.",
                "Reduzir estímulos e manter monitoramento sequencial da expressão de dor.",
                "Reavaliar após intervalo curto para comparar tendência multimodal.",
            ],
            "alertas": alertas,
            "resumo_tecnico": (
                "Análise de dor corporal (sem diagnóstico): "
                f"dor_max={pain_max:.2f}, "
                f"frames_sofrimento={suffering_frames}, "
                f"alteracoes_posturais={posture_changes}, "
                f"frames_sangue_facial={blood_frames}, "
                f"frames_hematoma_facial={bruise_frames}, "
                f"score_hematoma_max={max_bruise_score:.3f}, "
                f"anomalias_audio={len(anomalies)}."
            ),
            "priority": "high" if alertas else "moderate",
            "vision": vision_payload,
        }

    if normalized == CATEGORY_LAPARO:
        max_instrument_conf = max(
            (_safe_float(row.get("confidence"), 0.0) for row in instrument_rows if str(row.get("class_group", "")).strip().lower() == "instrumento"),
            default=0.0,
        )
        max_bleeding_conf = max(
            (_safe_float(row.get("confidence"), 0.0) for row in instrument_rows if str(row.get("class_group", "")).strip().lower() == "bleeding"),
            default=0.0,
        )
        bleeding_events = sum(
            1 for row in instrument_rows if str(row.get("class_group", "")).strip().lower() == "bleeding"
        )
        instrument_events = sum(
            1 for row in instrument_rows if str(row.get("class_group", "")).strip().lower() == "instrumento"
        )
        unknown_instruments = sum(
            1
            for row in instrument_rows
            if str(row.get("class_name", "")).strip().lower() == "instrumento cirúrgico não identificado"
        )
        contraction_conf = max(
            (
                _safe_float(row.get("temporal_action_conf"), 0.0)
                for row in events_rows
                if "contra" in str(row.get("temporal_action_label", "")).strip().lower()
            ),
            default=0.0,
        )
        evidence_boost = 0.0
        if bleeding_events >= 1:
            evidence_boost = max(evidence_boost, 0.35)
        if instrument_events >= 3:
            evidence_boost = max(evidence_boost, 0.35)
        if instrument_events >= 8:
            evidence_boost = max(evidence_boost, 0.45)
        confidence = max(max_instrument_conf, max_bleeding_conf, contraction_conf, evidence_boost)
        if confidence < MIN_CONFIDENCE_LAPARO:
            return {
                "categoria": CATEGORY_INDETERMINADO,
                "acoes_prioritarias": [
                    "Revisar frames internos com maior resolução e contraste.",
                    "Executar nova inferência para confirmar instrumento/contração.",
                ],
                "alertas": False,
                "resumo_tecnico": "Sem padrão interno com confiança mínima para laparoscopia após agregação temporal.",
                "priority": "low",
            }
        alertas = bool(bleeding_events > 0 or unknown_instruments > 0)
        return {
            "categoria": CATEGORY_LAPARO,
            "acoes_prioritarias": [
                "Validar instrumentos detectados e sequência temporal do procedimento.",
                "Inspecionar eventos de sangramento com prioridade quando persistentes.",
                "Sinalizar revisão imediata para instrumento cirúrgico não identificado.",
            ],
            "alertas": alertas,
            "resumo_tecnico": (
                "Análise laparoscópica interna: "
                f"conf_instrumento_max={max_instrument_conf:.3f}, "
                f"conf_sangramento_max={max_bleeding_conf:.3f}, "
                f"eventos_instrumento={instrument_events}, "
                f"eventos_sangramento={bleeding_events}, "
                f"instrumentos_nao_identificados={unknown_instruments}, "
                f"conf_contracao_max={contraction_conf:.3f}, "
                f"confianca_agregada={confidence:.3f}."
            ),
            "priority": "high" if alertas else "moderate",
        }

    return {
        "categoria": CATEGORY_INDETERMINADO,
        "acoes_prioritarias": [
            "Definir uma das três categorias obrigatórias antes do processamento.",
            "Revalidar configuração do caso na entrada.",
        ],
        "alertas": False,
        "resumo_tecnico": "Categoria do caso ausente ou fora das opções suportadas.",
        "priority": "low",
        "vision": vision_payload,
    }
