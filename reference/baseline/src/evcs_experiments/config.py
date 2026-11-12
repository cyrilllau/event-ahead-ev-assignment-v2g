"""Explicit project paths; configuration never changes scientific case inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tomllib
from typing import Mapping


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    spec_root: Path
    campaign_root: Path
    frozen_source_root: Path

    def as_dict(self):
        return {key: str(value) for key, value in asdict(self).items()}


def get_paths(config=None, project_root=None):
    """Read TOML/JSON or a mapping; relative values resolve against project root.

    Priority: explicit config, EVCS_CONFIG, configs/local.toml when present,
    built-in local defaults. No directory is created and no file is modified.
    """
    root = Path(project_root or os.environ.get("EVCS_PROJECT_ROOT") or
                Path(__file__).resolve().parents[2]).expanduser().resolve()
    selected = config if config is not None else os.environ.get("EVCS_CONFIG")
    if selected is None:
        default = root / "configs/local.toml"
        selected = default if default.exists() else {}
    if isinstance(selected, Mapping):
        settings = dict(selected)
    else:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = root / path
        with path.open("rb") as handle:
            settings = json.load(handle) if path.suffix.lower() == ".json" else tomllib.load(handle)
    values = settings.get("paths", settings)
    if not isinstance(values, Mapping):
        raise ValueError("Configuration paths must be a mapping")
    allowed = {"project_root", "data_root", "spec_root", "campaign_root", "frozen_source_root"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration path keys: {sorted(unknown)}")
    # An explicit argument/environment project root takes precedence; a file
    # may otherwise declare a root relative to the discovered project root.
    if "project_root" in values and project_root is None and not os.environ.get("EVCS_PROJECT_ROOT"):
        override = Path(values["project_root"]).expanduser()
        root = (override if override.is_absolute() else root / override).resolve()
    defaults = {"data_root": "data/cases/ieee33-v54-q1", "spec_root": "docs/experiments",
        "campaign_root": "artifacts/campaigns/ieee33-v54-q1", "frozen_source_root": "provenance/legacy_s01"}
    resolved = {}
    for name, default in defaults.items():
        value = values.get(name, default)
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f"{name} must be a nonempty path")
        path = Path(value).expanduser()
        resolved[name] = (path if path.is_absolute() else root / path).resolve()
    return ProjectPaths(root, **resolved)
