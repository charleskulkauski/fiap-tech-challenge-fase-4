#!/usr/bin/env python3
"""
Baixa modelos e datasets do Hugging Face para deploy local.

Repos:
  - charleskulkauski/fiap-gynecology-models
  - charleskulkauski/fiap-gynecology-datasets (zipados)

Uso:
  python scripts/download_hf_assets.py --models
  python scripts/download_hf_assets.py --datasets
  python scripts/download_hf_assets.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hf_assets import (  # noqa: E402
    DATA_ROOT,
    HF_DATASETS_REPO,
    HF_MODELS_REPO,
    HF_MODELS_ROOT,
    download_datasets,
    download_models,
    models_ready,
    temporal_models_ready,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa modelos (.pt) e datasets (zip) do Hugging Face.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models",
        action="store_true",
        help="Baixa apenas checkpoints (inferência).",
    )
    group.add_argument(
        "--datasets",
        action="store_true",
        help="Baixa e extrai datasets zipados (treino).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Baixa modelos e datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-baixa mesmo se arquivos já existirem.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    do_models = args.all or args.models or (not args.datasets and not args.all)
    do_datasets = args.all or args.datasets

    print(f"[download_hf] models repo:   {HF_MODELS_REPO}")
    print(f"[download_hf] datasets repo: {HF_DATASETS_REPO}")
    print(f"[download_hf] destino modelos: {HF_MODELS_ROOT}")
    print(f"[download_hf] destino dados:   {DATA_ROOT}")

    if do_models:
        paths = download_models(force=args.force)
        print(f"[download_hf] modelos baixados: {len(paths)} arquivo(s)")
        print(f"[download_hf] yolo pronto: {models_ready()}")
        print(f"[download_hf] temporal pronto: {temporal_models_ready()}")

    if do_datasets:
        paths = download_datasets(force=args.force)
        print(f"[download_hf] datasets extraídos: {len(paths)} pasta(s)")
        for path in paths:
            print(f"  - {path}")

    if not do_models and not do_datasets:
        print("Nada selecionado. Use --models, --datasets ou --all.", file=sys.stderr)
        return 1

    print("[download_hf] Concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
