from pathlib import Path

import pytest

from fed_policy_intelligence.settings import ConfigError, load_settings


def test_default_config_loads_from_project_root() -> None:
    root = Path(__file__).resolve().parents[1]

    settings = load_settings(project_root=root, environ={})

    assert settings.name == "federal-reserve-policy-intelligence"
    assert settings.runtime.device == "cpu"
    assert settings.llm.provider == "mock"
    assert settings.data.corpus_path == (root / "fomc_corpus.csv").resolve()


def test_environment_overrides_runtime_selections() -> None:
    root = Path(__file__).resolve().parents[1]

    settings = load_settings(
        project_root=root,
        environ={
            "FPI_DEVICE": "auto",
            "FPI_LOG_LEVEL": "debug",
            "FPI_LLM_PROVIDER": "mock",
        },
    )

    assert settings.runtime.device == "auto"
    assert settings.runtime.log_level == "DEBUG"


def test_invalid_stride_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "tests" / "fixtures" / "invalid_stride.yaml"

    with pytest.raises(ConfigError, match="stride"):
        load_settings(config, project_root=root, environ={})


def test_unknown_provider_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]

    with pytest.raises(ConfigError, match="provider"):
        load_settings(
            project_root=root,
            environ={"FPI_LLM_PROVIDER": "unconfigured-provider"},
        )

