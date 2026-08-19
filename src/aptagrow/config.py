"""Configuration loading and command-specific path validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class Config:
    """Loaded YAML configuration with repository-relative path resolution."""

    data: dict[str, Any]
    root: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def path(self, *keys: str, required: bool = False) -> Path:
        value = self.get(*keys)
        label = ".".join(keys)
        if value in (None, "") or (isinstance(value, str) and "${" in value):
            if required:
                raise ValueError(f"Configure '{label}' before running this stage")
            return Path("")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def require_files(self, items: Iterable[tuple[tuple[str, ...], bool]]) -> None:
        """Validate configured paths; the bool indicates directory vs. file."""
        missing: list[str] = []
        for keys, is_dir in items:
            path = self.path(*keys, required=True)
            exists = path.is_dir() if is_dir else path.is_file()
            if not exists:
                missing.append(f"{'.'.join(keys)} -> {path}")
        if missing:
            raise FileNotFoundError("Missing configured resources:\n- " + "\n- ".join(missing))


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        with parent_path.resolve().open("r", encoding="utf-8") as handle:
            base_data = yaml.safe_load(handle) or {}
        data = _merge(base_data, data)
    data = _expand(data)
    # The shipped configuration lives in <repo>/config/default.yaml.
    root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return Config(data=data, root=root.resolve())
