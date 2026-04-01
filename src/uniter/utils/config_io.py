from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from uniter.config import AppConfig


def write_resolved_config(config: AppConfig, destination: str | Path) -> Path:
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            asdict(config),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
