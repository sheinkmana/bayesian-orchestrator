from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "mode" not in config:
        raise ValueError("Config must include a 'mode' field.")

    config["_config_path"] = str(path.resolve())
    return config


def output_dir(config: dict[str, Any]) -> Path:
    configured = config.get("output_dir", "outputs/run")
    path = Path(configured)
    path.mkdir(parents=True, exist_ok=True)
    return path
