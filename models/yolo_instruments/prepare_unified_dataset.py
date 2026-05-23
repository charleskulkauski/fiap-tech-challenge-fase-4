
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    source_name: str
    target_class_id: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_instruments = (
        project_root / "data" / "medical_fiap-tech-challenge.yolov8"
    ).resolve()
    default_bleeding = (
        project_root / "data" / "bleeding_fiap-tech-challenge.yolov8"
    ).resolve()
    default_output = (project_root / "data" / "combined_medical_bleeding.yolov8").resolve()

    parser = argparse.ArgumentParser(
        description="Unifica datasets de instrumentos + bleeding para YOLOv8.",
    )
    parser.add_argument("--instruments-root", default=str(default_instruments))
    parser.add_argument("--bleeding-root", default=str(default_bleeding))
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--train-instrument-repeat",
        type=int,
        default=8,
        help=(
            "Repeticao de amostras de instrumento apenas no treino para "
            "reduzir desbalanceamento extremo."
        ),
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Falha ao encontrar linhas YOLO invalidas em vez de ignora-las.",
    )
    return parser.parse_args(argv)


def _validate_ratio(name: str, value: float) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} precisa estar no intervalo [0, 1). Recebido: {value}")


def _collect_samples(dataset_root: Path, source_name: str, target_class_id: int) -> list[Sample]:
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise FileNotFoundError(
            f"Estrutura invalida em {dataset_root}. Esperado train/images e train/labels."
        )

    labels = sorted(train_labels.glob("*.txt"))
    if not labels:
        raise RuntimeError(f"Nenhuma label encontrada em {train_labels}")

    samples: list[Sample] = []
    for label in labels:
        stem = label.stem
        image_path = None
        for ext in _IMAGE_EXTENSIONS:
            candidate = train_images / f"{stem}{ext}"
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            raise RuntimeError(f"Imagem correspondente ausente para label: {label}")
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label,
                source_name=source_name,
                target_class_id=target_class_id,
            )
        )

    return samples


def _validate_and_remap_yolo_lines(
    lines: Iterable[str],
    target_class_id: int,
    *,
    strict_labels: bool,
    issue_prefix: str,
    issues: list[str],
) -> list[str]:
    output_lines: list[str] = []

    def report_or_raise(message: str) -> None:
        if strict_labels:
            raise ValueError(message)
        issues.append(message)

    for line_idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            report_or_raise(
                f"{issue_prefix}: linha YOLO invalida {line_idx}: {stripped!r}"
            )
            continue

        try:
            float(parts[0])
        except ValueError:
            report_or_raise(
                f"{issue_prefix}: classe nao numerica na linha {line_idx}: {stripped!r}"
            )
            continue

        coords = []
        for value in parts[1:]:
            try:
                coords.append(float(value))
            except ValueError:
                report_or_raise(
                    f"{issue_prefix}: coordenada invalida na linha {line_idx}: {stripped!r}"
                )
                coords = []
                break
        if not coords:
            continue
        x_center, y_center, width, height = coords

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            report_or_raise(
                f"{issue_prefix}: centro fora de [0,1] na linha {line_idx}: {stripped!r}"
            )
            continue
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            report_or_raise(
                f"{issue_prefix}: largura/altura fora de (0,1] na linha {line_idx}: {stripped!r}"
            )
            continue

        output_lines.append(
            f"{target_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return output_lines


def _split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Sample]]:
    shuffled = samples[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_test = int(round(total * test_ratio))
    n_val = int(round(total * val_ratio))
    n_test = min(n_test, total)
    n_val = min(n_val, max(total - n_test, 0))
    n_train = max(total - n_val - n_test, 0)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _prepare_split_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def _write_sample(
    sample: Sample,
    split_name: str,
    output_root: Path,
    serial: int,
    *,
    strict_labels: bool,
    issues: list[str],
) -> tuple[str, int]:
    image_name = f"{sample.source_name}_{serial:06d}{sample.image_path.suffix.lower()}"
    label_name = f"{sample.source_name}_{serial:06d}.txt"
    out_image = output_root / split_name / "images" / image_name
    out_label = output_root / split_name / "labels" / label_name

    raw_lines = sample.label_path.read_text(encoding="utf-8").splitlines()
    mapped_lines = _validate_and_remap_yolo_lines(
        raw_lines,
        sample.target_class_id,
        strict_labels=strict_labels,
        issue_prefix=str(sample.label_path),
        issues=issues,
    )
    if not mapped_lines:
        return "", 0

    shutil.copy2(sample.image_path, out_image)
    out_label.write_text("\n".join(mapped_lines) + "\n", encoding="utf-8")
    return image_name, len(mapped_lines)


def _write_data_yaml(output_root: Path) -> Path:
    payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["instrumento", "bleeding"],
    }
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml


def _write_hyperparameter_notes(output_root: Path) -> Path:
    notes = output_root / "hyperparameter_tuning_recommendations.txt"
    notes.write_text(
        "\n".join(
            [
                "Sugestoes para reduzir overfitting no treino YOLOv8 (dataset desbalanceado):",
                "",
                "1) Inicializacao",
                "   - Comece com pesos pretreinados (ex.: yolov8s.pt).",
                "   - Considere freeze=10 nas primeiras camadas em cenarios com poucos instrumentos.",
                "",
                "2) Regularizacao e aug",
                "   - lr0=0.003 a 0.006; lrf=0.01",
                "   - weight_decay=0.001",
                "   - dropout=0.05 (quando disponivel na tarefa)",
                "   - mosaic=0.6 e close_mosaic=10",
                "   - hsv_s=0.5, hsv_v=0.4, degrees=5.0, scale=0.2, translate=0.1",
                "",
                "3) Desbalanceamento",
                "   - oversampling de instrumento no treino (este script usa --train-instrument-repeat).",
                "   - aumentar box=8.0 e cls=1.0 pode ajudar em classes pequenas.",
                "   - monitorar AP por classe e nao apenas mAP global.",
                "",
                "4) Comando base sugerido (ajuste conforme GPU):",
                "   yolo detect train model=yolov8s.pt data=data/combined_medical_bleeding.yolov8/data.yaml "
                "epochs=80 imgsz=640 batch=16 patience=20 cos_lr=True lr0=0.004 lrf=0.01 weight_decay=0.001 "
                "mosaic=0.6 close_mosaic=10 box=8.0 cls=1.0 dfl=1.5",
                "",
                "5) Validacao robusta",
                "   - Use validacao temporal em video e revise falsos positivos de bleeding com filtro espectral.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return notes


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    instruments_root = Path(args.instruments_root).expanduser().resolve()
    bleeding_root = Path(args.bleeding_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    _validate_ratio("val_ratio", args.val_ratio)
    _validate_ratio("test_ratio", args.test_ratio)
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio precisa ser menor que 1.0")
    if args.train_instrument_repeat < 1:
        raise ValueError("--train-instrument-repeat precisa ser >= 1")

    rng = random.Random(args.seed)
    instrument_samples = _collect_samples(
        instruments_root,
        source_name="instrument",
        target_class_id=0,
    )
    bleeding_samples = _collect_samples(
        bleeding_root,
        source_name="bleeding",
        target_class_id=1,
    )

    split_instrument = _split_samples(
        instrument_samples, args.val_ratio, args.test_ratio, rng
    )
    split_bleeding = _split_samples(bleeding_samples, args.val_ratio, args.test_ratio, rng)

    _prepare_split_dirs(output_root)

    serial = 0
    registry_rows: list[dict[str, str | int]] = []
    split_counter = {
        "train": {"images": 0, "labels": 0, "instrument": 0, "bleeding": 0},
        "val": {"images": 0, "labels": 0, "instrument": 0, "bleeding": 0},
        "test": {"images": 0, "labels": 0, "instrument": 0, "bleeding": 0},
    }
    issues: list[str] = []

    for split_name in ("train", "val", "test"):
        split_samples = split_instrument[split_name] + split_bleeding[split_name]
        rng.shuffle(split_samples)

        expanded_samples: list[Sample] = []
        for sample in split_samples:
            repeats = args.train_instrument_repeat if (
                split_name == "train" and sample.source_name == "instrument"
            ) else 1
            expanded_samples.extend([sample] * repeats)

        for sample in expanded_samples:
            serial += 1
            image_name, n_labels = _write_sample(
                sample,
                split_name,
                output_root,
                serial,
                strict_labels=bool(args.strict_labels),
                issues=issues,
            )
            if not image_name:
                continue

            split_counter[split_name]["images"] += 1
            split_counter[split_name]["labels"] += n_labels
            split_counter[split_name][sample.source_name] += 1
            registry_rows.append(
                {
                    "split": split_name,
                    "source": sample.source_name,
                    "image_name": image_name,
                    "n_labels": n_labels,
                }
            )

    registry_path = output_root / "dataset_registry.csv"
    with registry_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "source", "image_name", "n_labels"])
        writer.writeheader()
        writer.writerows(registry_rows)

    data_yaml = _write_data_yaml(output_root)
    tuning_notes = _write_hyperparameter_notes(output_root)
    issues_path = output_root / "label_validation_issues.txt"
    issues_path.write_text(
        "\n".join(issues) + ("\n" if issues else ""),
        encoding="utf-8",
    )

    print(f"[dataset] dataset unificado criado em: {output_root}")
    print(f"[dataset] data.yaml: {data_yaml}")
    print(f"[dataset] tuning notes: {tuning_notes}")
    print(f"[dataset] issues: {issues_path} ({len(issues)} ocorrencias)")
    for split_name in ("train", "val", "test"):
        stats = split_counter[split_name]
        print(
            "[dataset] "
            f"{split_name}: images={stats['images']} labels={stats['labels']} "
            f"instrument={stats['instrument']} bleeding={stats['bleeding']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
