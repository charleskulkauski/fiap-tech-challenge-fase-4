from __future__ import annotations

import json
import shutil
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from streamlit_mvp.utils.case_policy import (
    CATEGORY_ACOLHIMENTO,
    CATEGORY_LAPARO,
    get_pain_behavior_thresholds,
    build_structured_case_output,
    enforce_case_policy_outputs,
    get_case_policy,
    normalize_case_category,
)
from streamlit_mvp.utils.case_store import get_case_files, load_case_meta, save_case_status
from streamlit_mvp.utils.project import ensure_project_imports
from src.hf_assets import resolve_mvp_yolo_model as resolve_hf_yolo_model


def _resolve_mvp_yolo_model(case_category: str) -> str | None:
    resolved = resolve_hf_yolo_model(case_category == CATEGORY_LAPARO)
    return str(resolved) if resolved is not None else None


def _copy_generated_artifact(
    source: Path | None,
    target: Path,
    *,
    remove_source: bool = False,
) -> Path | None:
    if source is None or not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if remove_source:
        try:
            if source.resolve() != target.resolve():
                source.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _delete_matches(directory: Path, patterns: list[str], keep: set[Path] | None = None) -> None:
    keep_resolved = {item.resolve() for item in (keep or set()) if item.exists()}
    for pattern in patterns:
        for candidate in directory.glob(pattern):
            try:
                if candidate.resolve() in keep_resolved:
                    continue
                candidate.unlink(missing_ok=True)
            except OSError:
                continue


def _cleanup_report_artifacts(files, *, keep_report_json: bool = False) -> None:
    keep = {
        files.risk_alert_json,
        files.prontuario_json,
        files.report_pdf,
    }
    if keep_report_json:
        keep.add(files.report_json)
    _delete_matches(
        files.reports_dir,
        [
            "risk_alert_*.json",
            "prontuario_multimodal_*.json",
            "relatorio_*.pdf",
            "relatorio_*.json",
            "video_instrument_events_prontuario_*.csv",
        ],
        keep=keep,
    )


def _cleanup_mode_specific_outputs(
    *,
    has_video_input: bool,
    has_audio_input: bool,
    files,
) -> None:
    if not has_video_input:
        files.video_out.unlink(missing_ok=True)
        files.video_events_csv.unlink(missing_ok=True)
        files.video_instrument_csv.unlink(missing_ok=True)
    if not has_audio_input:
        files.transcript_txt.unlink(missing_ok=True)
        files.audio_analysis_json.unlink(missing_ok=True)


def _split_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _run_mvp_video_phase(args: Namespace) -> dict:
    import inspect

    try:
        from src.pipeline_video import run_pipeline, set_patient_visual_thresholds
    except ImportError:
        from src.pipeline_video import run_pipeline
        set_patient_visual_thresholds = None

    if set_patient_visual_thresholds is not None:
        set_patient_visual_thresholds(
        visual_blood_persist_threshold=float(
            getattr(args, "visual_blood_persist_threshold", 0.85)
        ),
        visual_blood_score_threshold=float(
            getattr(args, "visual_blood_score_threshold", 0.85)
        ),
        visual_bruise_score_threshold=float(
            getattr(args, "visual_bruise_score_threshold", 0.06)
        ),
        visual_bruise_min_pixels=int(getattr(args, "visual_bruise_min_pixels", 12)),
        )

    call_kwargs = {
        "video_path": args.input_video,
        "output_video_path": args.output_video,
        "output_csv_path": args.events_csv,
        "pain_every_n_frames": 3,
        "pain_scale": 0.5,
        "smooth_window": 10,
        "yolo_every_n_frames": args.yolo_every,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_imgsz": args.yolo_imgsz,
        "yolo_scale": args.yolo_scale,
        "yolo_model_path": args.yolo_model,
        "yolo_instrument_csv_path": args.detections_csv,
        "bleeding_class_names": _split_csv_arg(args.bleeding_classes),
        "instrument_class_names": _split_csv_arg(args.instrument_classes),
        "enable_spectral_filter": not args.disable_spectral_filter,
        "min_overlap_for_risk": args.risk_overlap,
        "spectral_min_rg_ratio": args.spectral_min_rg,
        "spectral_min_rb_ratio": args.spectral_min_rb,
        "spectral_min_red_mean": args.spectral_min_red,
        "spectral_min_pixels": args.spectral_min_pixels,
        "enable_patient_signals": bool(getattr(args, "enable_patient_signals", True)),
        "enable_instrument_detection": bool(
            getattr(args, "enable_instrument_detection", True)
        ),
        "highlight_visual_blood_boxes": bool(
            getattr(args, "highlight_visual_blood_boxes", False)
        ),
        "visual_blood_history_len": int(getattr(args, "visual_blood_history_len", 4)),
        "visual_blood_persist_threshold": float(
            getattr(args, "visual_blood_persist_threshold", 0.85)
        ),
        "visual_blood_score_threshold": float(
            getattr(args, "visual_blood_score_threshold", 0.85)
        ),
        "enable_bruise_detection": bool(
            getattr(args, "enable_bruise_detection", True)
        ),
        "instrument_min_confidence": float(
            getattr(args, "instrument_min_confidence", 0.35)
        ),
        "instrument_min_streak": int(getattr(args, "instrument_min_streak", 2)),
        "register_only_red_instruments": bool(
            getattr(args, "register_only_red_instruments", False)
        ),
    }
    supported = inspect.signature(run_pipeline).parameters
    filtered = {key: value for key, value in call_kwargs.items() if key in supported}
    return run_pipeline(**filtered)


def run_case_pipeline(
    case_id: str,
    save_json: bool = False,
    stage_callback: Callable[[str], None] | None = None,
) -> dict:
    ensure_project_imports()

    from main import (
        _export_unified_prontuario,
        _run_audio_phase,
        _run_report_phase,
    )
    from models.report_generator import aggregate_video_events

    files = get_case_files(case_id)
    case_meta = load_case_meta(case_id)
    from models.lgpd_compliance import clear_case_lgpd_context, set_case_lgpd_context

    set_case_lgpd_context(case_meta)
    try:
        return _run_case_pipeline_impl(
            case_id=case_id,
            save_json=save_json,
            stage_callback=stage_callback,
            files=files,
            case_meta=case_meta,
        )
    finally:
        clear_case_lgpd_context()


def _run_case_pipeline_impl(
    *,
    case_id: str,
    save_json: bool,
    stage_callback: Callable[[str], None] | None,
    files,
    case_meta: dict,
) -> dict:
    from main import (
        _export_unified_prontuario,
        _run_audio_phase,
        _run_report_phase,
    )
    from models.report_generator import aggregate_video_events

    from models.lgpd_compliance import (
        allows_azure_speech,
        assert_local_processing_allowed,
        run_local_audio_phase,
        LgpdProcessingBlockedError,
    )

    assert_local_processing_allowed(case_meta)
    case_category = normalize_case_category(case_meta.get("categoria_caso"))
    case_policy = get_case_policy(case_category)
    has_video_input = files.video_input.is_file()
    has_audio_input = files.audio_input.is_file()
    run_audio_stage = has_audio_input and case_policy.allow_audio_transcript
    if not has_video_input and not has_audio_input:
        raise FileNotFoundError("Nenhum insumo encontrado no caso (vídeo/áudio).")
    _cleanup_mode_specific_outputs(
        has_video_input=has_video_input,
        has_audio_input=run_audio_stage,
        files=files,
    )

    started_at = datetime.now()
    save_case_status(
        case_id,
        {
            "processing_done": False,
            "report_ready": False,
            "last_error": "",
            "step": "video" if has_video_input else ("audio" if run_audio_stage else "risk"),
        },
    )

    video_summary: dict | None = None
    if has_video_input:
        if stage_callback:
            stage_callback("video")
        yolo_model_path = _resolve_mvp_yolo_model(case_category)
        pain_visual = get_pain_behavior_thresholds(case_category)
        video_summary = _run_mvp_video_phase(
            Namespace(
                output_video=str(files.video_out),
                events_csv=str(files.video_events_csv),
                detections_csv=str(files.video_instrument_csv),
                input_video=str(files.video_input),
                yolo_every=1 if case_category == CATEGORY_LAPARO else 2,
                yolo_conf=0.35 if case_category == CATEGORY_LAPARO else 0.20,
                yolo_iou=0.40 if case_category == CATEGORY_LAPARO else 0.5,
                yolo_imgsz=1280 if case_category == CATEGORY_LAPARO else 1280,
                yolo_scale=1.0,
                yolo_model=yolo_model_path,
                bleeding_classes="bleeding,sangramento,blood",
                instrument_classes="instrumento,instrument,instrumentos",
                disable_spectral_filter=False,
                risk_overlap=0.15,
                spectral_min_rg=1.12,
                spectral_min_rb=1.10,
                spectral_min_red=55.0,
                spectral_min_pixels=120,
                enable_patient_signals=case_category != CATEGORY_LAPARO,
                enable_instrument_detection=bool(
                    case_policy.allow_instrument_identification
                ),
                highlight_visual_blood_boxes=bool(
                    case_category != CATEGORY_LAPARO
                ),
                visual_blood_history_len=4,
                visual_blood_persist_threshold=(
                    0.85
                    if case_category == CATEGORY_LAPARO
                    else pain_visual.blood_persist
                ),
                visual_blood_score_threshold=(
                    0.85
                    if case_category == CATEGORY_LAPARO
                    else pain_visual.blood_score
                ),
                visual_bruise_score_threshold=(
                    0.06
                    if case_category == CATEGORY_LAPARO
                    else pain_visual.bruise_score
                ),
                visual_bruise_min_pixels=(
                    12 if case_category == CATEGORY_LAPARO else pain_visual.bruise_min_pixels
                ),
                enable_bruise_detection=bool(case_policy.allow_bruise_detection),
                instrument_min_confidence=0.35 if case_category == CATEGORY_LAPARO else 0.10,
                instrument_min_streak=1 if case_category == CATEGORY_LAPARO else 2,
                register_only_red_instruments=bool(case_category == CATEGORY_LAPARO),
            )
        )

    save_case_status(case_id, {"step": "audio"})
    audio_phase: dict = {"transcript_path": None, "audio_analysis_path": None, "audio_analysis": None}
    if run_audio_stage:
        if stage_callback:
            stage_callback("audio")
        if allows_azure_speech(case_meta):
            try:
                audio_phase = _run_audio_phase(
                    files.audio_input,
                    files.outputs_dir,
                    files.outputs_dir,
                    started_at,
                )
            except LgpdProcessingBlockedError:
                audio_phase = run_local_audio_phase(
                    files.audio_input,
                    files.outputs_dir,
                    started_at,
                )
        else:
            audio_phase = run_local_audio_phase(
                files.audio_input,
                files.outputs_dir,
                started_at,
            )
    transcript_path = (
        Path(audio_phase["transcript_path"])
        if audio_phase.get("transcript_path")
        else None
    )
    audio_analysis_path = (
        Path(audio_phase["audio_analysis_path"])
        if audio_phase.get("audio_analysis_path")
        else None
    )

    transcript_canonical = _copy_generated_artifact(
        transcript_path,
        files.transcript_txt,
        remove_source=True,
    )
    audio_analysis_canonical = _copy_generated_artifact(
        audio_analysis_path,
        files.audio_analysis_json,
        remove_source=True,
    )
    transcript_path = transcript_canonical or transcript_path
    audio_analysis_path = audio_analysis_canonical or audio_analysis_path
    _delete_matches(
        files.outputs_dir,
        ["transcript_*.txt", "audio_analysis_*.json"],
        keep={files.transcript_txt, files.audio_analysis_json},
    )

    if has_video_input:
        enforce_case_policy_outputs(
            category=case_category,
            events_csv=files.video_events_csv,
            instrument_csv=files.video_instrument_csv,
        )

    save_case_status(case_id, {"step": "risk"})
    if stage_callback:
        stage_callback("risk")
    final_risk_alert = build_structured_case_output(
        category=case_category,
        events_csv=files.video_events_csv,
        instrument_csv=files.video_instrument_csv,
        audio_analysis=audio_phase.get("audio_analysis"),
    )
    files.risk_alert_json.write_text(
        json.dumps(final_risk_alert, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_case_status(case_id, {"step": "prontuario"})
    if stage_callback:
        stage_callback("prontuario")
    prontuario_outputs = _export_unified_prontuario(
        instruments_csv=files.video_instrument_csv,
        output_dir=files.reports_dir,
        consultation_start=started_at,
        audio_analysis=audio_phase.get("audio_analysis"),
        final_risk_alert=final_risk_alert,
    )
    _copy_generated_artifact(
        prontuario_outputs.get("consolidated_json_path"),
        files.prontuario_json,
        remove_source=True,
    )
    if prontuario_outputs.get("consolidated_json_path") is None:
        files.prontuario_json.unlink(missing_ok=True)

    save_case_status(case_id, {"step": "report"})
    if stage_callback:
        stage_callback("report")
    _run_report_phase(
        transcript_path=transcript_path,
        events_csv=files.video_events_csv if files.video_events_csv.is_file() else None,
        instruments_csv=files.video_instrument_csv if files.video_instrument_csv.is_file() else None,
        reports_dir=files.reports_dir,
        consultation_start=started_at,
        case_category=case_category,
    )
    report_pdf_path = files.report_pdf

    json_path = None
    if save_json:
        fusion_summary = (
            aggregate_video_events(
                events_csv_path=files.video_events_csv,
                instruments_csv_path=files.video_instrument_csv,
            )
            if files.video_events_csv.is_file()
            else None
        )
        payload = {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "case_id": case_id,
            "inputs": {
                "video": str(files.video_input) if has_video_input else None,
                "audio": str(files.audio_input) if has_audio_input else None,
            },
            "outputs": {
                "video_out": str(files.video_out) if files.video_out.is_file() else None,
                "video_events_csv": str(files.video_events_csv) if files.video_events_csv.is_file() else None,
                "video_instrument_events_csv": None,
                "transcript_txt": str(files.transcript_txt) if files.transcript_txt.is_file() else None,
                "audio_analysis_json": str(files.audio_analysis_json) if files.audio_analysis_json.is_file() else None,
                "risk_alert_json": str(files.risk_alert_json),
                "prontuario_json": str(files.prontuario_json) if files.prontuario_json.is_file() else None,
                "report_pdf": str(files.report_pdf),
            },
            "final_risk_alert": final_risk_alert,
            "video_pipeline": video_summary,
            "video_aggregate_summary": asdict(fusion_summary) if fusion_summary is not None else None,
        }
        files.report_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_path = files.report_json

    _cleanup_report_artifacts(files, keep_report_json=save_json)

    save_case_status(
        case_id,
        {
            "processing_done": True,
            "report_ready": True,
            "case_closed": False,
            "last_error": "",
            "step": "done",
            "last_run_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    if stage_callback:
        stage_callback("done")

    return {
        "video_summary": video_summary,
        "report_pdf": Path(report_pdf_path),
        "report_json": json_path,
        "transcript_len": len(files.transcript_txt.read_text(encoding="utf-8")) if files.transcript_txt.is_file() else 0,
        "final_risk_alert": final_risk_alert,
        "risk_alert_json": files.risk_alert_json if files.risk_alert_json.is_file() else None,
        "audio_analysis_json": files.audio_analysis_json if files.audio_analysis_json.is_file() else None,
        "prontuario_json": files.prontuario_json if files.prontuario_json.is_file() else None,
    }
