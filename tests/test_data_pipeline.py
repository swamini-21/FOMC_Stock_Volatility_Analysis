from pathlib import Path

import pandas as pd

from fed_policy_intelligence.data import pipeline
from fed_policy_intelligence.settings import (
    DataSettings,
    LlmSettings,
    ModelSettings,
    ProjectSettings,
    RuntimeSettings,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_pipeline_prepares_output_payloads(monkeypatch) -> None:
    output_dir = Path("ignored-test-output")
    settings = ProjectSettings(
        name="test-project",
        root_dir=FIXTURES.parent,
        config_path=Path("test-config.yaml"),
        data=DataSettings(
            corpus_path=FIXTURES / "fed_sample.csv",
            corpus_source="test corpus",
            vix_path=FIXTURES / "vix_sample.csv",
            market_source="test market",
            market_series_id="VIXCLS",
            processing_version="test",
            processed_dir=output_dir,
            cache_dir=Path("ignored-cache"),
            artifacts_dir=Path("ignored-artifacts"),
        ),
        runtime=RuntimeSettings(seed=42, device="cpu", log_level="INFO"),
        models=ModelSettings(
            sentiment_model="sentiment/model",
            emotion_model="emotion/model",
            max_length=128,
            stride=16,
            batch_size=2,
        ),
        llm=LlmSettings(provider="mock"),
    )
    csv_writes: list[tuple[pd.DataFrame, Path]] = []
    json_writes: list[tuple[dict[str, object], Path]] = []
    monkeypatch.setattr(
        pipeline,
        "_write_csv_atomic",
        lambda frame, destination: csv_writes.append((frame, destination)),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_json_atomic",
        lambda payload, destination: json_writes.append((payload, destination)),
    )

    result = pipeline.run_data_pipeline(settings)

    assert len(csv_writes) == 2
    assert len(json_writes) == 1
    assert json_writes[0][0]["documents"]["output_rows"] == 3
    assert json_writes[0][0]["market"]["usable_rows"] == 2
    assert json_writes[0][0]["inputs"]["market"]["series_id"] == "VIXCLS"
    assert result.output_paths["documents"] == output_dir / "fed_documents.csv"

