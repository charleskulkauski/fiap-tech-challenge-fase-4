"""
Resolução de paths e bootstrap de modelos/datasets hospedados no Hugging Face.

Repos padrão:
  - charleskulkauski/fiap-gynecology-models
  - charleskulkauski/fiap-gynecology-datasets
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_MODELS_ROOT = Path(
    os.environ.get("HF_MODELS_ROOT", str(PROJECT_ROOT / "hf_models"))
).expanduser()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()

HF_MODELS_REPO = os.environ.get(
    "HF_MODELS_REPO", "charleskulkauski/fiap-gynecology-models"
)
HF_DATASETS_REPO = os.environ.get(
    "HF_DATASETS_REPO", "charleskulkauski/fiap-gynecology-datasets"
)

DEPLOYMENT_DIR = HF_MODELS_ROOT / "deployment"
TEMPORAL_EVENTS_ROOT = (
    HF_MODELS_ROOT / "action_events" / "gynsurg_action_3sec_round2"
)
LEGACY_WEIGHTS_DIR = PROJECT_ROOT / "models" / "yolo_instruments" / "weights"
LEGACY_TEMPORAL_ROOT = (
    PROJECT_ROOT / "models" / "action_events" / "runs" / "gynsurg_action_3sec_round2"
)

# zip no repo HF -> pasta relativa em data/
DATASET_ARCHIVES: dict[str, str] = {
    "gynsurg.zip": "gynsurg",
    "medical_fiap-tech-challenge.yolov8.zip": "medical_fiap-tech-challenge.yolov8",
    "bleeding_fiap-tech-challenge.yolov8.zip": "bleeding_fiap-tech-challenge.yolov8",
    "combined_medical_bleeding.yolov8.zip": "combined_medical_bleeding.yolov8",
    "gynsurg_instruments_detection.yolov8.zip": "gynsurg_instruments_detection.yolov8",
}

# Arquivos mínimos para inferência (deploy)
REQUIRED_MODEL_FILES: tuple[str, ...] = (
    "deployment/best.pt",
    "deployment/best_combined.pt",
    "deployment/best_instrument_bleeding.pt",
    "action_events/gynsurg_action_3sec_round2/action/best.pt",
    "action_events/gynsurg_action_3sec_round2/bleeding/best.pt",
    "action_events/gynsurg_action_3sec_round2/smoke/best.pt",
)


def _is_valid_pt(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if _is_valid_pt(candidate):
            return candidate
    return None


def deployment_weights_dir() -> Path:
    return DEPLOYMENT_DIR


def resolve_yolo_weight(*names: str) -> Path | None:
    """Resolve peso YOLO por nome(s) de arquivo, priorizando hf_models/deployment."""
    candidates: list[Path] = []
    for name in names:
        candidates.append(DEPLOYMENT_DIR / name)
        candidates.append(LEGACY_WEIGHTS_DIR / name)
    return _first_existing(candidates)


def resolve_default_yolo_model() -> Path | None:
    return resolve_yolo_weight("best.pt")


def resolve_laparoscopy_yolo_model() -> Path | None:
    return _first_existing(
        [
            DEPLOYMENT_DIR / "best_instrument_bleeding.pt",
            DEPLOYMENT_DIR / "best_combined.pt",
            DEPLOYMENT_DIR / "best.pt",
            LEGACY_WEIGHTS_DIR / "best_instrument_bleeding.pt",
            LEGACY_WEIGHTS_DIR / "best_combined.pt",
            LEGACY_WEIGHTS_DIR / "best.pt",
            HF_MODELS_ROOT / "yolo" / "gynsurg_instruments_detection" / "best.pt",
            HF_MODELS_ROOT / "yolo" / "combined_medical_bleeding" / "best.pt",
        ]
    )


def resolve_mvp_yolo_model(case_category_laparo: bool) -> Path | None:
    if case_category_laparo:
        return resolve_laparoscopy_yolo_model()
    return resolve_default_yolo_model()


def resolve_temporal_events_root() -> Path:
    if TEMPORAL_EVENTS_ROOT.is_dir() and any(
        (TEMPORAL_EVENTS_ROOT / task / "best.pt").is_file()
        for task in ("action", "bleeding", "smoke")
    ):
        return TEMPORAL_EVENTS_ROOT
    return LEGACY_TEMPORAL_ROOT


def models_ready() -> bool:
    return resolve_default_yolo_model() is not None


def temporal_models_ready() -> bool:
    root = resolve_temporal_events_root()
    return all((root / task / "best.pt").is_file() for task in ("action", "bleeding", "smoke"))


def _hf_hub_download(repo_id: str, filename: str, *, repo_type: str = "model") -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    return Path(local)


def _copy_into_project(relative_path: str, cached_file: Path) -> Path:
    target = HF_MODELS_ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if cached_file.resolve() != target.resolve():
        target.write_bytes(cached_file.read_bytes())
    return target


def download_models(*, force: bool = False) -> list[Path]:
    """Baixa checkpoints do repo fiap-gynecology-models para hf_models/."""
    downloaded: list[Path] = []
    for rel in REQUIRED_MODEL_FILES:
        target = HF_MODELS_ROOT / rel
        if not force and _is_valid_pt(target):
            downloaded.append(target)
            continue
        cached = _hf_hub_download(HF_MODELS_REPO, rel, repo_type="model")
        downloaded.append(_copy_into_project(rel, cached))
    _sync_legacy_deployment_aliases()
    return downloaded


def _sync_legacy_deployment_aliases() -> None:
    """Copia aliases de deploy para models/yolo_instruments/weights (compatibilidade)."""
    if not DEPLOYMENT_DIR.is_dir():
        return
    LEGACY_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "best_combined.pt", "best_instrument_bleeding.pt"):
        source = DEPLOYMENT_DIR / name
        if _is_valid_pt(source):
            target = LEGACY_WEIGHTS_DIR / name
            if not _is_valid_pt(target):
                target.write_bytes(source.read_bytes())


def _dataset_marker(dataset_dir: Path) -> Path:
    return dataset_dir / ".hf_download_ok"


def _extract_zip_to_data(zip_path: Path, dataset_rel: str) -> Path:
    target_root = DATA_ROOT / dataset_rel
    target_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Se o zip contém uma pasta raiz igual ao destino, extrai o conteúdo interno.
        prefix = f"{dataset_rel}/"
        if any(n.startswith(prefix) for n in names):
            for member in names:
                if member.endswith("/"):
                    continue
                if not member.startswith(prefix):
                    continue
                rel = member[len(prefix) :]
                if not rel:
                    continue
                out = target_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))
        else:
            zf.extractall(target_root)

    _fix_data_yaml_paths(target_root)
    _dataset_marker(target_root).write_text("ok\n", encoding="utf-8")
    return target_root


def _fix_data_yaml_paths(dataset_dir: Path) -> None:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return
    text = data_yaml.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("path:"):
            lines.append("path: .")
        else:
            lines.append(line)
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_datasets(*, force: bool = False) -> list[Path]:
    """Baixa e extrai datasets zipados do repo fiap-gynecology-datasets."""
    extracted: list[Path] = []
    cache_dir = HF_MODELS_ROOT / "_hf_cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for archive, dataset_rel in DATASET_ARCHIVES.items():
        target_root = DATA_ROOT / dataset_rel
        marker = _dataset_marker(target_root)
        if not force and marker.is_file() and target_root.is_dir():
            extracted.append(target_root)
            continue

        cached = _hf_hub_download(archive, archive, repo_type="dataset")
        zip_copy = cache_dir / archive
        if Path(cached).resolve() != zip_copy.resolve():
            zip_copy.write_bytes(Path(cached).read_bytes())
        extracted.append(_extract_zip_to_data(zip_copy, dataset_rel))

    return extracted


def ensure_models(*, force: bool = False) -> bool:
    if not force and models_ready():
        _sync_legacy_deployment_aliases()
        return True
    if os.environ.get("SKIP_HF_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}:
        return models_ready()
    try:
        download_models(force=force)
        return models_ready()
    except Exception as exc:
        print(f"[hf_assets] Aviso: falha ao baixar modelos ({exc}).", flush=True)
        return models_ready()


def ensure_datasets(*, force: bool = False) -> bool:
    if os.environ.get("SKIP_HF_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        download_datasets(force=force)
        return True
    except Exception as exc:
        print(f"[hf_assets] Aviso: falha ao baixar datasets ({exc}).", flush=True)
        return False


def load_manifest() -> dict:
    manifest_path = HF_MODELS_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
