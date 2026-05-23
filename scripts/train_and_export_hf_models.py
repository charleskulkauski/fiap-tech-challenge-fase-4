"""
Treina todos os modelos do projeto (YOLO + eventos temporais) e exporta
checkpoints organizados em hf_models/ para upload no Hugging Face.

Uso:
    python scripts/train_and_export_hf_models.py
    python scripts/train_and_export_hf_models.py --skip-yolo
    python scripts/train_and_export_hf_models.py --only yolo_combined,yolo_gynsurg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hf_assets import HF_MODELS_ROOT as _HF_MODELS_ROOT

HF_MODELS_ROOT = _HF_MODELS_ROOT
YOLO_RUNS_ROOT = PROJECT_ROOT / "models" / "yolo_instruments" / "runs"
ACTION_EXPORT_ROOT = HF_MODELS_ROOT / "action_events" / "gynsurg_action_3sec_round2"


@dataclass(frozen=True)
class YoloTrainJob:
    key: str
    data_yaml: Path
    run_name: str
    export_subdir: str
    deployment_alias: str | None
    epochs: int
    imgsz: int
    batch: int
    patience: int
    skip_class_validation: bool = False


YOLO_JOBS: tuple[YoloTrainJob, ...] = (
    YoloTrainJob(
        key="yolo_medical",
        data_yaml=PROJECT_ROOT / "data/medical_fiap-tech-challenge.yolov8/data.yaml",
        run_name="hf_medical_instruments",
        export_subdir="yolo/medical_fiap-tech-challenge",
        deployment_alias="best.pt",
        epochs=40,
        imgsz=320,
        batch=8,
        patience=15,
    ),
    YoloTrainJob(
        key="yolo_bleeding",
        data_yaml=PROJECT_ROOT / "data/bleeding_fiap-tech-challenge.yolov8/data.yaml",
        run_name="hf_bleeding",
        export_subdir="yolo/bleeding_fiap-tech-challenge",
        deployment_alias=None,
        epochs=12,
        imgsz=320,
        batch=8,
        patience=6,
        skip_class_validation=True,
    ),
    YoloTrainJob(
        key="yolo_combined",
        data_yaml=PROJECT_ROOT / "data/combined_medical_bleeding.yolov8/data.yaml",
        run_name="hf_combined_instrument_bleeding",
        export_subdir="yolo/combined_medical_bleeding",
        deployment_alias="best_combined.pt",
        epochs=12,
        imgsz=384,
        batch=4,
        patience=6,
        skip_class_validation=True,
    ),
    YoloTrainJob(
        key="yolo_gynsurg",
        data_yaml=PROJECT_ROOT / "data/gynsurg_instruments_detection.yolov8/data.yaml",
        run_name="hf_gynsurg_instruments",
        export_subdir="yolo/gynsurg_instruments_detection",
        deployment_alias="best_instrument_bleeding.pt",
        epochs=5,
        imgsz=512,
        batch=8,
        patience=5,
        skip_class_validation=True,
    ),
)


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n[hf_export] Executando: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _export_yolo_checkpoint(job: YoloTrainJob) -> Path:
    source = YOLO_RUNS_ROOT / job.run_name / "weights" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint YOLO não encontrado: {source}")

    export_dir = HF_MODELS_ROOT / job.export_subdir
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / "best.pt"
    shutil.copy2(source, target)
    print(f"[hf_export] YOLO exportado: {target}", flush=True)

    if job.deployment_alias:
        deploy_dir = HF_MODELS_ROOT / "deployment"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deploy_target = deploy_dir / job.deployment_alias
        shutil.copy2(source, deploy_target)
        print(f"[hf_export] Alias deploy: {deploy_target}", flush=True)

    return target


def train_yolo_job(job: YoloTrainJob, *, device: str) -> Path:
    if not job.data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml ausente: {job.data_yaml}")

    cmd = [
        sys.executable,
        "-m",
        "models.yolo_instruments.train",
        "--data",
        str(job.data_yaml),
        "--model",
        "yolov8n.pt",
        "--epochs",
        str(job.epochs),
        "--imgsz",
        str(job.imgsz),
        "--batch",
        str(job.batch),
        "--patience",
        str(job.patience),
        "--device",
        device,
        "--project",
        str(YOLO_RUNS_ROOT),
        "--name",
        job.run_name,
        "--no-copy-best",
    ]
    if job.skip_class_validation:
        cmd.append("--skip-class-validation")

    _run(cmd)
    return _export_yolo_checkpoint(job)


def train_action_events(
    *,
    device_note: str,
    epochs: int,
    max_samples_per_class: int,
) -> None:
    action_root = (
        PROJECT_ROOT
        / "data"
        / "gynsurg"
        / "GynSurg_Action_3sec"
        / "GynSurg_Action_3sec"
    )
    if not action_root.is_dir():
        raise FileNotFoundError(f"Raiz GynSurg action ausente: {action_root}")

    ACTION_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "models.action_events.train_gynsurg_action",
        "--action-root",
        str(action_root),
        "--output-root",
        str(ACTION_EXPORT_ROOT),
        "--epochs",
        str(epochs),
        "--batch-size",
        "4",
        "--max-samples-per-class",
        str(max_samples_per_class),
        "--use-balanced-sampler",
        "--use-class-weighted-loss",
    ]
    _run(cmd)

    for task in ("action", "bleeding", "smoke"):
        ckpt = ACTION_EXPORT_ROOT / task / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint temporal ausente: {ckpt}")
        print(f"[hf_export] Temporal OK: {ckpt} ({ckpt.stat().st_size} bytes)", flush=True)

    print(f"[hf_export] action_events device={device_note}", flush=True)


def _write_manifest(
    *,
    device: str,
    action_epochs: int,
    action_max_samples: int,
    completed: list[str],
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "structure": {
            "yolo": "hf_models/yolo/<dataset>/best.pt",
            "deployment": "hf_models/deployment/{best.pt,best_combined.pt,best_instrument_bleeding.pt}",
            "action_events": "hf_models/action_events/gynsurg_action_3sec_round2/{action,bleeding,smoke}/best.pt",
        },
        "yolo_jobs": [
            {**asdict(job), "data_yaml": str(job.data_yaml)}
            for job in YOLO_JOBS
        ],
        "action_events": {
            "epochs": action_epochs,
            "max_samples_per_class": action_max_samples,
            "note": (
                "max_samples_per_class limita clips por classe em CPU; "
                "remova o limite para treino completo."
            ),
        },
        "completed_steps": completed,
    }
    HF_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = HF_MODELS_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[hf_export] manifest: {manifest_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e exporta checkpoints para hf_models/.")
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device Ultralytics/PyTorch (ex.: "cpu", "0"). Default: cpu.',
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-action", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Lista separada por virgula: yolo_medical,yolo_bleeding,yolo_combined,yolo_gynsurg,action",
    )
    parser.add_argument("--action-epochs", type=int, default=5)
    parser.add_argument(
        "--action-max-samples-per-class",
        type=int,
        default=80,
        help="Limite de clips/classe para treino temporal viavel em CPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    completed: list[str] = []

    try:
        import torch  # noqa: F401
    except ImportError:
        print("Erro: PyTorch não instalado.", file=sys.stderr)
        return 1

    device = args.device
    if device != "cpu":
        import torch

        if not torch.cuda.is_available():
            print("[hf_export] CUDA indisponível; usando cpu.", flush=True)
            device = "cpu"

    HF_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_yolo:
        for job in YOLO_JOBS:
            if only and job.key not in only:
                continue
            train_yolo_job(job, device=device)
            completed.append(job.key)

    if not args.skip_action and (not only or "action" in only):
        train_action_events(
            device_note=device,
            epochs=args.action_epochs,
            max_samples_per_class=args.action_max_samples_per_class,
        )
        completed.append("action_events")

    _write_manifest(
        device=device,
        action_epochs=args.action_epochs,
        action_max_samples=args.action_max_samples_per_class,
        completed=completed,
    )
    print("\n[hf_export] Concluído. Checkpoints em hf_models/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
