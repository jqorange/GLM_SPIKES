from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def _split_model_string(model_str: str) -> List[str]:
    model_str = str(model_str).strip()
    if not model_str:
        return []
    return [s for s in model_str.replace(",", "_").split("_") if s]


def load_forward_selected_models(session_dir: Path) -> Dict[int, List[str]]:
    selected_csv = Path(session_dir) / "selected_models.csv"
    if not selected_csv.exists():
        return {}

    out: Dict[int, List[str]] = {}
    with open(selected_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            neuron_name = str(row.get("neuron", "")).strip()
            model_str = str(row.get("final_model", "")).strip()
            if not neuron_name:
                continue
            if not model_str:
                continue
            if not neuron_name.lower().startswith("neuron_"):
                continue
            try:
                idx = int(neuron_name.split("_", maxsplit=1)[1]) - 1
            except Exception:
                continue
            if idx < 0:
                continue
            model_vars = _split_model_string(model_str)
            if model_vars:
                out[idx] = model_vars
    return out
