from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    name: str
    random_state: int
    output_dir: str
    model_dir: str
    log_level: str


@dataclass
class DataConfig:
    path: str
    format: str
    text_column: str
    label_column: str
    required_columns: list[str]
    preprocess: Dict[str, Any]
    resources: Dict[str, str]
    output: Dict[str, str]
    source_text_column: str | None = None
    source_label_column: str | None = None
    read_kwargs: Dict[str, Any] | None = None


@dataclass
class ModelConfig:
    token_matching: Dict[str, Any]
    equipment_based: Dict[str, Any]
    semantic_similarity: Dict[str, Any]
    umec: Dict[str, Any]
    large_dataset_safety: Dict[str, Any] | None = None


@dataclass
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    models: ModelConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path) -> AppConfig:
    config_dir = Path(config_dir)
    project_cfg = _load_yaml(config_dir / "project.yaml").get("project", {})
    data_cfg = _load_yaml(config_dir / "data.yaml").get("data", {})
    model_cfg = _load_yaml(config_dir / "model.yaml").get("models", {})

    return AppConfig(
        project=ProjectConfig(**project_cfg),
        data=DataConfig(**data_cfg),
        models=ModelConfig(**model_cfg),
    )
