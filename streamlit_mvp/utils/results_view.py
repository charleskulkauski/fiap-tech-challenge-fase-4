from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

from streamlit_mvp.utils.case_policy import (
    EVENTS_CSV_COLUMNS_PT_TO_EN,
    get_pain_behavior_thresholds,
)
from streamlit_mvp.utils.project import ensure_project_imports

KEYWORDS = [
    "dor",
    "desconforto",
    "ansiedade",
    "medo",
    "violência",
    "abuso",
    "sangramento",
    "depressão",
    "hesitação",
    "risco",
    "alerta",
    "urgente",
]


def load_summary_from_csv(
    events_csv: Path,
    instrument_csv: Path,
    *,
    include_hematoma: bool = True,
    category: str | None = None,
) -> dict:
    ensure_project_imports()
    from models.report_generator import aggregate_video_events

    summary = aggregate_video_events(
        events_csv_path=events_csv,
        instruments_csv_path=instrument_csv,
    )
    # Regra dedicada para acolhimento/dor:
    # usa apenas sinais visuais no video_events.csv (sem lógica de instrumento/laparo).
    events_df = pd.read_csv(events_csv)
    events_df = events_df.rename(columns=EVENTS_CSV_COLUMNS_PT_TO_EN)
    blood_flag_col = (
        "flag_sangue_identificado"
        if "flag_sangue_identificado" in events_df.columns
        else "visual_blood_detected"
        if "visual_blood_detected" in events_df.columns
        else None
    )
    blood_score_col = (
        "score_sangue_identificado"
        if "score_sangue_identificado" in events_df.columns
        else "visual_blood_score"
        if "visual_blood_score" in events_df.columns
        else None
    )
    blood_flag_frames = 0
    max_blood_score = 0.0
    if blood_flag_col is not None:
        blood_flag_frames = int(
            pd.to_numeric(events_df[blood_flag_col], errors="coerce")
            .fillna(0.0)
            .astype(int)
            .clip(lower=0)
            .sum()
        )
    if blood_score_col is not None:
        max_blood_score = float(
            pd.to_numeric(events_df[blood_score_col], errors="coerce")
            .fillna(0.0)
            .max()
        )
    t = get_pain_behavior_thresholds(category)
    sangue_identificado = (
        "Sim"
        if (
            blood_flag_frames >= t.summary_min_blood_frames
            or max_blood_score >= t.summary_min_blood_score
        )
        else "Não"
    )
    hematoma_identificado = "Não"
    if include_hematoma:
        bruise_flag_col = (
            "flag_equimose_identificada"
            if "flag_equimose_identificada" in events_df.columns
            else "flag_hematoma_identificado"
            if "flag_hematoma_identificado" in events_df.columns
            else None
        )
        bruise_score_col = (
            "score_equimose_identificada"
            if "score_equimose_identificada" in events_df.columns
            else "score_hematoma_identificado"
            if "score_hematoma_identificado" in events_df.columns
            else None
        )
        bruise_flag_frames = 0
        max_bruise_score = 0.0
        if bruise_flag_col is not None:
            bruise_flag_frames = int(
                pd.to_numeric(events_df[bruise_flag_col], errors="coerce")
                .fillna(0.0)
                .astype(int)
                .clip(lower=0)
                .sum()
            )
        if bruise_score_col is not None:
            max_bruise_score = float(
                pd.to_numeric(events_df[bruise_score_col], errors="coerce")
                .fillna(0.0)
                .max()
            )
        hematoma_identificado = (
            "Sim"
            if (
                bruise_flag_frames >= t.summary_min_bruise_frames
                or max_bruise_score >= t.summary_min_bruise_score
            )
            else "Não"
        )
    return {
        "pain_smoothed_max": summary.pain_smoothed_max or 0.0,
        "pain_peaks_count": len(summary.pain_peaks),
        "n_instruments": len(summary.instrument_counts),
        "flags": summary.flag_rates,
        "sangue_identificado": sangue_identificado,
        "hematoma_identificado": hematoma_identificado,
    }


def build_top_events_table(
    events_csv: Path,
    instrument_csv: Path | None = None,
    top_n: int = 15,
    *,
    include_hematoma: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(events_csv)
    df = df.rename(columns=EVENTS_CSV_COLUMNS_PT_TO_EN)
    if df.empty:
        return df
    if "frame_idx" in df.columns:
        df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    else:
        df["frame_idx"] = pd.Series(range(len(df)), dtype="int64")
    if "score_sangue_identificado" not in df.columns and "visual_blood_score" in df.columns:
        df["score_sangue_identificado"] = df["visual_blood_score"]
    if "flag_sangue_identificado" not in df.columns and "visual_blood_detected" in df.columns:
        df["flag_sangue_identificado"] = df["visual_blood_detected"]
    if "score_equimose_identificada" not in df.columns and "score_hematoma_identificado" in df.columns:
        df["score_equimose_identificada"] = df["score_hematoma_identificado"]
    if "flag_equimose_identificada" not in df.columns and "flag_hematoma_identificado" in df.columns:
        df["flag_equimose_identificada"] = df["flag_hematoma_identificado"]

    if instrument_csv is not None and instrument_csv.is_file():
        try:
            inst_df = pd.read_csv(instrument_csv)
        except Exception:
            inst_df = pd.DataFrame()
        if not inst_df.empty and "frame_idx" in inst_df.columns:
            inst_df["frame_idx"] = pd.to_numeric(inst_df["frame_idx"], errors="coerce")
            inst_df["confidence"] = pd.to_numeric(
                inst_df.get("confidence", 0.0), errors="coerce"
            ).fillna(0.0)
            inst_df["class_group"] = (
                inst_df.get("class_group", "")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            inst_df["risk_alert"] = pd.to_numeric(
                inst_df.get("risk_alert", 0), errors="coerce"
            ).fillna(0).astype(int)

            grouped_rows: list[dict[str, object]] = []
            for frame_idx, frame_group in inst_df.groupby("frame_idx", dropna=True):
                instrument_group = frame_group[frame_group["class_group"] == "instrumento"]
                bleeding_group = frame_group[frame_group["class_group"] == "bleeding"]
                instrument_names = (
                    instrument_group.get("class_name", pd.Series(dtype="object"))
                    .astype(str)
                    .str.strip()
                )
                instrument_names = sorted({name for name in instrument_names if name})
                grouped_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "inst_n_instruments": int(len(instrument_names)),
                        "inst_instruments_seen": "|".join(instrument_names),
                        "inst_bleeding_score": float(
                            bleeding_group["confidence"].max()
                            if not bleeding_group.empty
                            else 0.0
                        ),
                        "inst_bleeding_detected": int(
                            (not bleeding_group.empty)
                            or bool((frame_group["risk_alert"] == 1).any())
                        ),
                    }
                )
            if grouped_rows:
                inst_frame_df = pd.DataFrame(grouped_rows)
                df = df.merge(inst_frame_df, on="frame_idx", how="left")
                if "n_instruments" in df.columns:
                    n_inst = pd.to_numeric(df["n_instruments"], errors="coerce").fillna(0.0)
                    n_inst_from_det = pd.to_numeric(
                        df["inst_n_instruments"], errors="coerce"
                    ).fillna(0.0)
                    df["n_instruments"] = n_inst.where(
                        n_inst > 0.0, n_inst_from_det
                    )
                if "instruments_seen" in df.columns:
                    raw_seen = df["instruments_seen"].fillna("").astype(str)
                    det_seen = df["inst_instruments_seen"].fillna("").astype(str)
                    df["instruments_seen"] = raw_seen.where(
                        raw_seen.str.strip() != "", det_seen
                    )
                if "score_sangue_identificado" in df.columns:
                    blood_evt = pd.to_numeric(
                        df["score_sangue_identificado"], errors="coerce"
                    ).fillna(0.0)
                    blood_det = pd.to_numeric(
                        df["inst_bleeding_score"], errors="coerce"
                    ).fillna(0.0)
                    df["score_sangue_identificado"] = blood_evt.where(
                        blood_evt > blood_det, blood_det
                    )
                if "flag_sangue_identificado" in df.columns:
                    blood_evt_flag = pd.to_numeric(
                        df["flag_sangue_identificado"], errors="coerce"
                    ).fillna(0).astype(int)
                    blood_det_flag = pd.to_numeric(
                        df["inst_bleeding_detected"], errors="coerce"
                    ).fillna(0).astype(int)
                    df["flag_sangue_identificado"] = blood_evt_flag.where(
                        blood_evt_flag > blood_det_flag, blood_det_flag
                    )

    cols = [
        "timestamp_sec",
        "pain_smoothed",
        "behavior_score",
        "n_instruments",
        "instruments_seen",
        "flag_curved_posture",
        "flag_asymmetric_shoulders",
        "visual_aggression_persisted",
        "score_sangue_identificado",
        "flag_sangue_identificado",
    ]
    if include_hematoma:
        cols.extend(
            [
                "score_equimose_identificada",
                "flag_equimose_identificada",
            ]
        )
    existing = [c for c in cols if c in df.columns]
    rank_df = df.copy()

    def _col_or_zeros(dataframe: pd.DataFrame, column: str) -> pd.Series:
        if column in dataframe.columns:
            return pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=dataframe.index, dtype="float64")

    for c in (
        "pain_smoothed",
        "behavior_score",
        "n_instruments",
        "visual_aggression_persisted",
        "score_sangue_identificado",
        *(
            ("score_equimose_identificada",)
            if include_hematoma
            else ()
        ),
    ):
        if c in rank_df.columns:
            rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce").fillna(0.0)
    rank_df["severity_rank"] = (
        _col_or_zeros(rank_df, "pain_smoothed") * 0.6
        + _col_or_zeros(rank_df, "behavior_score") * 0.3
        + _col_or_zeros(rank_df, "n_instruments") * 10.0 * 0.1
        + _col_or_zeros(rank_df, "visual_aggression_persisted") * 20.0 * 0.1
        + (
            _col_or_zeros(rank_df, "score_equimose_identificada") * 12.0 * 0.1
            if include_hematoma
            else 0.0
        )
    )
    rank_df = rank_df.sort_values("severity_rank", ascending=False)
    top_df = rank_df[existing + ["severity_rank"]].head(top_n).reset_index(drop=True)

    def _faixa_dor(value: float) -> str:
        if value >= 40.0:
            return "Alta (dor severa)"
        if value >= 25.0:
            return "Moderada (desconforto)"
        return "Baixa/estável"

    def _faixa_postural(value: float) -> str:
        if value >= 60.0:
            return "Alta alteração postural"
        if value >= 30.0:
            return "Alteração postural moderada"
        return "Baixa alteração postural"

    def _faixa_severidade(value: float) -> str:
        if value >= 45.0:
            return "Alta prioridade"
        if value >= 25.0:
            return "Prioridade moderada"
        return "Baixa prioridade"

    top_df["faixa_dor"] = _col_or_zeros(top_df, "pain_smoothed").apply(_faixa_dor)
    top_df["faixa_postural"] = (
        _col_or_zeros(top_df, "behavior_score").apply(_faixa_postural)
    )
    top_df["faixa_prioridade_evento"] = (
        _col_or_zeros(top_df, "severity_rank").apply(_faixa_severidade)
    )
    top_df["flag_sangue_identificado"] = (
        _col_or_zeros(top_df, "flag_sangue_identificado")
        .astype(int)
        .apply(lambda value: "Sim" if int(value) == 1 else "Não")
    )
    if include_hematoma:
        top_df["flag_equimose_identificada"] = (
            _col_or_zeros(top_df, "flag_equimose_identificada")
            .astype(int)
            .apply(lambda value: "Sim" if int(value) == 1 else "Não")
        )

    readable_columns = {
        "timestamp_sec": "tempo_video_s",
        "pain_smoothed": "dor_suavizada_pct",
        "behavior_score": "score_postural",
        "n_instruments": "qtd_instrumentos",
        "instruments_seen": "instrumentos_detectados",
        "flag_curved_posture": "flag_postura_curvada",
        "flag_asymmetric_shoulders": "flag_assimetria_ombros",
        "visual_aggression_persisted": "score_agressao_visual_persistido",
        "score_sangue_identificado": "score_sangue_identificado",
        "flag_sangue_identificado": "flag_sangue_identificado",
        "faixa_dor": "faixa_operacional_dor",
        "faixa_postural": "faixa_operacional_postura",
        "faixa_prioridade_evento": "faixa_prioridade_evento",
    }
    if include_hematoma:
        readable_columns.update(
            {
                "score_equimose_identificada": "score_hematoma_identificado",
                "flag_equimose_identificada": "flag_hematoma_identificado",
            }
        )
    renamed = top_df.rename(columns=readable_columns)
    available_columns = [col for col in readable_columns.values() if col in renamed.columns]
    return renamed[available_columns]


def highlight_keywords(text: str) -> str:
    escaped = html.escape(text or "")
    for kw in KEYWORDS:
        pattern = re.compile(re.escape(kw), flags=re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f"<mark>{m.group(0)}</mark>",
            escaped,
        )
    return escaped.replace("\n", "<br>")
