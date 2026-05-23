
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


@dataclass(frozen=True)
class Sample:
    source_name: str
    group_key: str
    image_path: Path
    mask_path: Path | None


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepara dataset YOLO (deteccao) a partir das mascaras GynSurg.",
    )
    parser.add_argument(
        "--gynsurg-root",
        default=str(project_root / "data" / "gynsurg"),
        help="Raiz contendo GynSurg_Instrument_Dataset, Auxiliary e Anatomy.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root / "data" / "gynsurg_instruments_detection.yolov8"),
        help="Pasta de saida no formato YOLOv8 detect.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=24,
        help="Area minima (px) de componente conectado para virar bbox.",
    )
    parser.add_argument(
        "--include-anatomy-as-background",
        action="store_true",
        help="Inclui frames de Anatomy com labels vazias como negativos.",
    )
    return parser.parse_args()


def _iter_png_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.png") if p.is_file()])


def _collect_unique_values(mask_root: Path) -> list[int]:
    values: set[int] = set()
    for mask_path in _iter_png_files(mask_root):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        for v in set(mask.reshape(-1).tolist()):
            if int(v) != 0:
                values.add(int(v))
    return sorted(values)


def _build_class_map(gynsurg_root: Path) -> tuple[dict[tuple[str, int], int], list[str]]:
    sources = [
        (
            "instrument",
            gynsurg_root
            / "GynSurg_Instrument_Dataset"
            / "GynSurg_Instrument_Dataset"
            / "insseg_mask",
        ),
        (
            "auxiliary",
            gynsurg_root
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "tool_mask",
        ),
    ]
    class_map: dict[tuple[str, int], int] = {}
    class_names: list[str] = []
    class_id = 0
    for source_name, mask_root in sources:
        values = _collect_unique_values(mask_root)
        for value in values:
            class_map[(source_name, value)] = class_id
            class_names.append(f"{source_name}_v{value}")
            class_id += 1
    return class_map, class_names


def _build_samples(gynsurg_root: Path, include_anatomy_bg: bool) -> list[Sample]:
    samples: list[Sample] = []
    specs = [
        (
            "instrument",
            gynsurg_root
            / "GynSurg_Instrument_Dataset"
            / "GynSurg_Instrument_Dataset"
            / "insseg",
            gynsurg_root
            / "GynSurg_Instrument_Dataset"
            / "GynSurg_Instrument_Dataset"
            / "insseg_mask",
        ),
        (
            "auxiliary",
            gynsurg_root
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "tool",
            gynsurg_root
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "GynSurg_Auxiliary_Tool_Dataset"
            / "tool_mask",
        ),
    ]
    if include_anatomy_bg:
        specs.append(
            (
                "anatomy_bg",
                gynsurg_root
                / "GynSurg_Anatomy_Dataset"
                / "GynSurg_Anatomy_Dataset"
                / "ganseg",
                None,
            )
        )

    for source_name, image_root, mask_root in specs:
        image_files = _iter_png_files(image_root)
        if not image_files:
            raise RuntimeError(f"Nenhuma imagem encontrada em {image_root}")
        for image_path in image_files:
            rel = image_path.relative_to(image_root)
            group_key = rel.parent.as_posix()
            if mask_root is None:
                sample = Sample(source_name, group_key, image_path, None)
            else:
                mask_rel = Path(str(rel).replace(".png", "_mask.png"))
                mask_path = mask_root / mask_rel
                if not mask_path.is_file():
                    raise RuntimeError(f"Mascara ausente para imagem: {image_path}")
                sample = Sample(source_name, group_key, image_path, mask_path)
            samples.append(sample)
    return samples


def _split_groups(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Sample]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio e test_ratio invalidos.")

    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        key = f"{sample.source_name}:{sample.group_key}"
        groups.setdefault(key, []).append(sample)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    n_groups = len(group_items)
    n_test = int(round(n_groups * test_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_test = min(n_test, n_groups)
    n_val = min(n_val, max(0, n_groups - n_test))

    test_groups = group_items[:n_test]
    val_groups = group_items[n_test : n_test + n_val]
    train_groups = group_items[n_test + n_val :]

    result = {"train": [], "val": [], "test": []}
    for split_name, split_groups in (
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ):
        for _, split_samples in split_groups:
            result[split_name].extend(split_samples)
    return result


def _mask_to_yolo_labels(
    mask_path: Path,
    source_name: str,
    class_map: dict[tuple[str, int], int],
    min_component_area: int,
) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []

    height, width = mask.shape[:2]
    labels: list[str] = []
    for value in sorted(set(mask.reshape(-1).tolist())):
        value = int(value)
        if value == 0:
            continue
        class_id = class_map.get((source_name, value))
        if class_id is None:
            continue
        binary = (mask == value).astype("uint8")
        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_idx in range(1, num_labels):
            x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            y = int(stats[comp_idx, cv2.CC_STAT_TOP])
            w = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < min_component_area or w < 2 or h < 2:
                continue
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )
    return labels


def _prepare_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def _write_data_yaml(output_root: Path, class_names: list[str]) -> Path:
    payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml


def main() -> int:
    args = _parse_args()
    gynsurg_root = Path(args.gynsurg_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    class_map, class_names = _build_class_map(gynsurg_root)
    samples = _build_samples(gynsurg_root, args.include_anatomy_as_background)
    splits = _split_groups(samples, args.val_ratio, args.test_ratio, args.seed)
    _prepare_dirs(output_root)

    serial = 0
    stats = {
        "train": {"images": 0, "labels": 0},
        "val": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0},
    }

    for split_name in ("train", "val", "test"):
        for sample in splits[split_name]:
            serial += 1
            stem = f"{sample.source_name}_{serial:06d}"
            image_out = output_root / split_name / "images" / f"{stem}.png"
            label_out = output_root / split_name / "labels" / f"{stem}.txt"
            shutil.copy2(sample.image_path, image_out)
            if sample.mask_path is None:
                label_lines: list[str] = []
            else:
                label_lines = _mask_to_yolo_labels(
                    sample.mask_path,
                    sample.source_name,
                    class_map,
                    args.min_component_area,
                )
            label_out.write_text(
                ("\n".join(label_lines) + "\n") if label_lines else "",
                encoding="utf-8",
            )
            stats[split_name]["images"] += 1
            stats[split_name]["labels"] += len(label_lines)

    data_yaml = _write_data_yaml(output_root, class_names)
    print(f"[gynsurg] dataset YOLO criado em: {output_root}")
    print(f"[gynsurg] data.yaml: {data_yaml}")
    print(f"[gynsurg] classes ({len(class_names)}): {class_names}")
    for split_name in ("train", "val", "test"):
        print(
            f"[gynsurg] {split_name}: "
            f"images={stats[split_name]['images']} "
            f"labels={stats[split_name]['labels']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
