from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DEMO_ROOT = PROJECT_ROOT / "data" / "demo"


def ensure_project_imports() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def normalize_case_id(raw_case_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", (raw_case_id or "").strip())
    clean = re.sub(r"-{2,}", "-", clean).strip("-")
    return clean.lower()


def case_root(case_id: str) -> Path:
    return DATA_DEMO_ROOT / case_id


def case_inputs_dir(case_id: str) -> Path:
    return case_root(case_id) / "inputs"


def case_outputs_dir(case_id: str) -> Path:
    return case_root(case_id) / "outputs"


def case_reports_dir(case_id: str) -> Path:
    return case_root(case_id) / "reports"
