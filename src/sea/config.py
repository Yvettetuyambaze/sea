from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["output_dir"])


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=raw, path=path)
