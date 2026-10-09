"""Validated project settings loaded from YAML and environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


@dataclass(frozen=True)
class DataSettings:
    """Locations for source data and generated artifacts."""

    corpus_path: Path
    vix_path: Path
    processed_dir: Path
    cache_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Execution settings shared by local CPU and Colab runs."""

    seed: int
    device: str
    log_level: str


@dataclass(frozen=True)
class ModelSettings:
    """Model identifiers and inference limits."""

    sentiment_model: str
    emotion_model: str
    max_length: int
    stride: int
    batch_size: int


@dataclass(frozen=True)
class LlmSettings:
    """LLM provider selection; Phase 2 uses the offline mock provider."""

    provider: str


@dataclass(frozen=True)
class ProjectSettings:
    """Complete validated configuration for one project run."""

    name: str
    root_dir: Path
    config_path: Path
    data: DataSettings
    runtime: RuntimeSettings
    models: ModelSettings
    llm: LlmSettings


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section '{section}' must be a mapping.")
    return value


def _required(section: Mapping[str, Any], key: str, section_name: str) -> Any:
    if key not in section or section[key] is None:
        raise ConfigError(f"Missing required configuration value: {section_name}.{key}")
    return section[key]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a positive integer.") from exc
    if parsed <= 0:
        raise ConfigError(f"{field} must be a positive integer.")
    return parsed


def _path(root_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root_dir / path).resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectSettings:
    """Load and validate settings.

    Environment variables override runtime selections only. Paths remain in the
    versioned YAML file so a run can be reconstructed from its saved config.
    """

    env = os.environ if environ is None else environ
    root_dir = Path(project_root).resolve() if project_root else _project_root()

    selected_path = config_path or env.get("FPI_CONFIG_PATH", "config/config.yaml")
    candidate = Path(selected_path).expanduser()
    resolved_config = (
        candidate.resolve() if candidate.is_absolute() else (root_dir / candidate).resolve()
    )
    if not resolved_config.is_file():
        raise ConfigError(f"Configuration file not found: {resolved_config}")

    try:
        raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file: {resolved_config}") from exc

    document = _mapping(raw, "root")
    project = _mapping(_required(document, "project", "root"), "project")
    data = _mapping(_required(document, "data", "root"), "data")
    runtime = _mapping(_required(document, "runtime", "root"), "runtime")
    models = _mapping(_required(document, "models", "root"), "models")
    llm = _mapping(_required(document, "llm", "root"), "llm")

    name = str(_required(project, "name", "project")).strip()
    if not name:
        raise ConfigError("project.name must not be empty.")

    seed = _positive_int(_required(runtime, "seed", "runtime"), "runtime.seed")
    device = env.get("FPI_DEVICE", str(_required(runtime, "device", "runtime"))).lower()
    if device not in {"cpu", "cuda", "auto"}:
        raise ConfigError("runtime.device must be one of: cpu, cuda, auto.")

    log_level = env.get(
        "FPI_LOG_LEVEL", str(_required(runtime, "log_level", "runtime"))
    ).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("runtime.log_level is not a supported Python logging level.")

    max_length = _positive_int(
        _required(models, "max_length", "models"), "models.max_length"
    )
    stride = _positive_int(_required(models, "stride", "models"), "models.stride")
    if stride >= max_length - 2:
        raise ConfigError("models.stride must leave room for content and special tokens.")

    sentiment_model = str(
        _required(models, "sentiment_model", "models")
    ).strip()
    emotion_model = str(_required(models, "emotion_model", "models")).strip()
    if not sentiment_model or not emotion_model:
        raise ConfigError("Model identifiers must not be empty.")

    provider = env.get("FPI_LLM_PROVIDER", str(_required(llm, "provider", "llm"))).lower()
    if provider not in {"mock", "openai", "anthropic"}:
        raise ConfigError("llm.provider must be one of: mock, openai, anthropic.")

    return ProjectSettings(
        name=name,
        root_dir=root_dir,
        config_path=resolved_config,
        data=DataSettings(
            corpus_path=_path(
                root_dir, _required(data, "corpus_path", "data"), "data.corpus_path"
            ),
            vix_path=_path(root_dir, _required(data, "vix_path", "data"), "data.vix_path"),
            processed_dir=_path(
                root_dir, _required(data, "processed_dir", "data"), "data.processed_dir"
            ),
            cache_dir=_path(
                root_dir, _required(data, "cache_dir", "data"), "data.cache_dir"
            ),
            artifacts_dir=_path(
                root_dir, _required(data, "artifacts_dir", "data"), "data.artifacts_dir"
            ),
        ),
        runtime=RuntimeSettings(seed=seed, device=device, log_level=log_level),
        models=ModelSettings(
            sentiment_model=sentiment_model,
            emotion_model=emotion_model,
            max_length=max_length,
            stride=stride,
            batch_size=_positive_int(
                _required(models, "batch_size", "models"), "models.batch_size"
            ),
        ),
        llm=LlmSettings(provider=provider),
    )

