from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from streamlit_mvp.utils.project import (
    case_inputs_dir,
    case_outputs_dir,
    case_reports_dir,
    case_root,
)
from streamlit_mvp.utils.validators import AUDIO_EXTENSIONS

CASE_META_FILE = "case_meta.json"
CASE_STATUS_FILE = "status.json"


@dataclass
class CaseFiles:
    case_id: str
    root: Path
    inputs_dir: Path
    outputs_dir: Path
    reports_dir: Path
    video_input: Path
    audio_input: Path
    video_out: Path
    video_events_csv: Path
    video_instrument_csv: Path
    transcript_txt: Path
    report_pdf: Path
    report_json: Path
    audio_analysis_json: Path
    risk_alert_json: Path
    prontuario_json: Path
    case_meta: Path
    status_file: Path


def _default_status() -> dict:
    return {
        "case_created": False,
        "processing_done": False,
        "report_ready": False,
        "case_closed": False,
        "last_error": "",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _resolve_audio_input_path(inputs_dir: Path) -> Path:
    candidates: list[Path] = []
    for ext in sorted(AUDIO_EXTENSIONS):
        candidate = inputs_dir / f"audio_input{ext}"
        if candidate.is_file():
            candidates.append(candidate)

    if not candidates:
        return inputs_dir / "audio_input.wav"

    return max(candidates, key=lambda item: item.stat().st_mtime)


def get_case_files(case_id: str) -> CaseFiles:
    root = case_root(case_id)
    inputs_dir = case_inputs_dir(case_id)
    outputs_dir = case_outputs_dir(case_id)
    reports_dir = case_reports_dir(case_id)
    return CaseFiles(
        case_id=case_id,
        root=root,
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        reports_dir=reports_dir,
        video_input=inputs_dir / "video_input.mp4",
        audio_input=_resolve_audio_input_path(inputs_dir),
        video_out=outputs_dir / "video_out.mp4",
        video_events_csv=outputs_dir / "video_events.csv",
        video_instrument_csv=outputs_dir / "video_instrument_events.csv",
        transcript_txt=outputs_dir / "transcript.txt",
        report_pdf=reports_dir / "relatorio.pdf",
        report_json=reports_dir / "relatorio.json",
        audio_analysis_json=outputs_dir / "audio_analysis.json",
        risk_alert_json=reports_dir / "risk_alert.json",
        prontuario_json=reports_dir / "prontuario_multimodal.json",
        case_meta=root / CASE_META_FILE,
        status_file=root / CASE_STATUS_FILE,
    )


def ensure_case_dirs(case_id: str) -> CaseFiles:
    files = get_case_files(case_id)
    files.inputs_dir.mkdir(parents=True, exist_ok=True)
    files.outputs_dir.mkdir(parents=True, exist_ok=True)
    files.reports_dir.mkdir(parents=True, exist_ok=True)
    if not files.status_file.is_file():
        files.status_file.write_text(
            json.dumps(_default_status(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return files


def save_case_meta(case_id: str, payload: dict) -> None:
    files = ensure_case_dirs(case_id)
    files.case_meta.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_case_meta(case_id: str) -> dict:
    files = get_case_files(case_id)
    if not files.case_meta.is_file():
        return {}
    return json.loads(files.case_meta.read_text(encoding="utf-8"))


def load_case_status(case_id: str) -> dict:
    files = get_case_files(case_id)
    if not files.status_file.is_file():
        return _default_status()
    return json.loads(files.status_file.read_text(encoding="utf-8"))


def save_case_status(case_id: str, status: dict) -> None:
    files = ensure_case_dirs(case_id)
    payload = load_case_status(case_id)
    payload.update(status)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    files.status_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
