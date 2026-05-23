
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml

_MODULE_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _MODULE_DIR / "weights"
_DEFAULT_REQUIRED_CLASSES = (
    "Bend hemostat",
    "Curved tip surgical scissors",
    "Dressing forceps",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina YOLOv8 com Ultralytics para instrumentos médicos e copia "
            "o best.pt para models/yolo_instruments/weights/best.pt."
        ),
    )
    parser.add_argument(
        "--data",
        help=(
            "Caminho para o data.yaml do dataset YOLOv8 (vindo do Roboflow "
            "ou de outro export). Se não for informado, use os parâmetros "
            "--roboflow-* para baixar automaticamente."
        ),
    )
    parser.add_argument(
        "--roboflow-workspace",
        help="Slug do workspace no Roboflow",
    )
    parser.add_argument(
        "--roboflow-project",
        help="Slug do projeto no Roboflow",
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        help="Versão do dataset no Roboflow",
    )
    parser.add_argument(
        "--roboflow-format",
        default="yolov8",
        help='Formato de export do Roboflow. Default: "yolov8".',
    )
    parser.add_argument(
        "--roboflow-api-key-env",
        default="ROBOFLOW_API_KEY",
        help=(
            "Nome da variável de ambiente com a API key do Roboflow. "
            'Default: "ROBOFLOW_API_KEY".'
        ),
    )
    parser.add_argument(
        "--required-classes",
        default=",".join(_DEFAULT_REQUIRED_CLASSES),
        help=(
            "Classes obrigatórias no dataset, separadas por vírgula. "
            "Default: Bend hemostat, Curved tip surgical scissors, "
            "Dressing forceps."
        ),
    )
    parser.add_argument(
        "--skip-class-validation",
        action="store_true",
        help="Ignora validação das classes obrigatórias no data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Peso inicial (yolov8n.pt, yolov8s.pt, ...). Default: yolov8s.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Número de epochs. Default: 80.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolução de entrada. Default: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Default: 16.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device do PyTorch ("cpu", "0", "cuda:0"). Default: auto.',
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience. Default: 30.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.004,
        help="Learning rate inicial. Default: 0.004.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Learning rate final (fator). Default: 0.01.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay para reduzir overfitting. Default: 0.001.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.6,
        help="Probabilidade de mosaic augmentation. Default: 0.6.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=15,
        help="Desativa mosaic nas ultimas N epochs. Default: 15.",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=8.0,
        help="Peso da loss de box. Default: 8.0.",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=1.0,
        help="Peso da loss de classificacao. Default: 1.0.",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=1.5,
        help="Peso da loss DFL. Default: 1.5.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Numero de camadas congeladas no backbone. Default: 0.",
    )
    parser.add_argument(
        "--no-cos-lr",
        action="store_true",
        help="Desabilita scheduler cosseno de LR (habilitado por padrao).",
    )
    parser.add_argument(
        "--project",
        default=str(_MODULE_DIR / "runs"),
        help=(
            "Pasta-raiz dos runs do Ultralytics. Default: "
            "models/yolo_instruments/runs."
        ),
    )
    parser.add_argument(
        "--name",
        default="train",
        help="Nome do experimento dentro de --project. Default: train.",
    )
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help=(
            "Não copia best.pt para models/yolo_instruments/weights/best.pt "
            "ao final do treino."
        ),
    )
    return parser.parse_args(argv)


def _normalize_label(label: str) -> str:
    lowered = label.strip().lower()
    return re.sub(r"[\s_-]+", " ", lowered)


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_roboflow_data_yaml(args: argparse.Namespace) -> Path:
    api_key = os.environ.get(args.roboflow_api_key_env)
    if not api_key:
        raise RuntimeError(
            "API key do Roboflow ausente. "
            f'Defina a variável de ambiente "{args.roboflow_api_key_env}".'
        )

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "Pacote 'roboflow' não encontrado. Instale com: pip install roboflow"
        ) from exc

    print(
        "[train] baixando dataset do Roboflow: "
        f"{args.roboflow_workspace}/{args.roboflow_project} v{args.roboflow_version}"
    )
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(args.roboflow_workspace)
        .project(args.roboflow_project)
        .version(args.roboflow_version)
        .download(args.roboflow_format)
    )
    data_path = (Path(dataset.location).resolve() / "data.yaml").resolve()
    if not data_path.is_file():
        raise RuntimeError(
            "Download do Roboflow concluído, mas data.yaml não foi encontrado "
            f"em {data_path}."
        )
    return data_path


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.data:
        return Path(args.data).expanduser().resolve()

    required_rf_args = (
        args.roboflow_workspace,
        args.roboflow_project,
        args.roboflow_version,
    )
    if all(required_rf_args):
        return _resolve_roboflow_data_yaml(args)

    raise RuntimeError(
        "Informe --data path/para/data.yaml OU todos os parâmetros "
        "--roboflow-workspace, --roboflow-project e --roboflow-version."
    )


def _read_dataset_class_names(data_path: Path) -> list[str]:
    parsed = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = parsed.get("names", [])

    if isinstance(names, dict):
        ordered_keys = sorted(names.keys(), key=lambda k: int(k))
        return [str(names[k]).strip() for k in ordered_keys if str(names[k]).strip()]

    if isinstance(names, list):
        return [str(name).strip() for name in names if str(name).strip()]

    return []


def _validate_required_classes(data_path: Path, required_classes: Iterable[str]) -> None:
    dataset_classes = _read_dataset_class_names(data_path)
    if not dataset_classes:
        raise RuntimeError(
            f"data.yaml sem classes válidas em 'names': {data_path}"
        )

    normalized_dataset = {_normalize_label(name) for name in dataset_classes}
    missing = [
        required
        for required in required_classes
        if _normalize_label(required) not in normalized_dataset
    ]
    if missing:
        raise RuntimeError(
            "Dataset incompatível com os instrumentos exigidos. "
            f"Faltando: {missing}. Classes encontradas: {dataset_classes}"
        )

    print(f"[train] classes do dataset validadas: {dataset_classes}")


def _copy_best_to_weights_dir(best_pt: Path) -> Path:
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = _WEIGHTS_DIR / "best.pt"
    shutil.copy2(best_pt, target)
    return target


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        data_path = _resolve_data_yaml(args)
    except RuntimeError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if not data_path.is_file():
        print(f"Erro: data.yaml não encontrado em {data_path}", file=sys.stderr)
        return 1

    required_classes = _parse_csv_list(args.required_classes)
    if required_classes and not args.skip_class_validation:
        try:
            _validate_required_classes(data_path, required_classes)
        except RuntimeError as exc:
            print(f"Erro: {exc}", file=sys.stderr)
            return 1

    from ultralytics import YOLO

    print(f"[train] base model: {args.model}")
    print(f"[train] data:       {data_path}")
    print(f"[train] epochs:     {args.epochs}")
    print(f"[train] imgsz:      {args.imgsz}")
    print("[train] tuning anti-overfitting ativado para dataset desbalanceado.")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "patience": args.patience,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "freeze": args.freeze,
        "cos_lr": not args.no_cos_lr,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", "") or "")
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        print(
            f"[train] AVISO: não encontrei {best_pt}. "
            "Verifique a pasta de runs.",
            file=sys.stderr,
        )
        return 0

    print(f"[train] best.pt em: {best_pt}")
    if not args.no_copy_best:
        copied = _copy_best_to_weights_dir(best_pt)
        print(f"[train] best.pt copiado para: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
