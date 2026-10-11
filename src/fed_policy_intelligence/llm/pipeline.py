"""Run cached structured policy extraction with the configured provider."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from fed_policy_intelligence.llm.extractor import PolicyIntelligenceExtractor
from fed_policy_intelligence.llm.gemini import GeminiPolicyProvider
from fed_policy_intelligence.llm.prompt import PROMPT_VERSION
from fed_policy_intelligence.llm.provider import MockPolicyProvider
from fed_policy_intelligence.llm.schema import SCHEMA_VERSION
from fed_policy_intelligence.settings import ProjectSettings, load_settings


def _load_cache(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    cached: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
            key = (
                str(record["document_id"]),
                str(record["source_sha256"]),
                str(record["provider"]),
                str(record["model_id"]),
                str(record["prompt_version"]),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Invalid extraction cache record on line {line_number}.") from exc
        cached[key] = record
    return cached


def run_extraction(
    documents: pd.DataFrame,
    *,
    extractor: PolicyIntelligenceExtractor,
    cache_path: Path,
    checkpoint_every: int | None = None,
    progress: Callable[[int, int, dict[str, Any], bool], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive when provided.")
    required = {"document_id", "cleaned_text"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"Document data is missing required columns: {missing}")
    cache = _load_cache(cache_path)
    records: list[dict[str, Any]] = []
    reused = 0
    total = len(documents)
    for position, row in enumerate(documents.itertuples(index=False), start=1):
        document_id = str(row.document_id)
        source_text = str(row.cleaned_text)
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        key = (
            document_id,
            source_hash,
            extractor.provider.provider_name,
            extractor.provider.model_id,
            PROMPT_VERSION,
        )
        cached = cache.get(key)
        if (
            cached is not None
            and cached.get("schema_version") == SCHEMA_VERSION
            and cached.get("status") in {"success", "invalid_evidence"}
        ):
            records.append(cached)
            reused += 1
            was_reused = True
        else:
            records.append(
                extractor.extract(document_id=document_id, source_text=source_text).to_dict()
            )
            was_reused = False
        if checkpoint_every is not None and (
            position % checkpoint_every == 0 or position == total
        ):
            _write_jsonl_atomic(records, cache_path)
        if progress is not None:
            progress(position, total, records[-1], was_reused)
    return records, reused


def _write_jsonl_atomic(records: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(destination)


def build_extractor(settings: ProjectSettings) -> PolicyIntelligenceExtractor:
    if settings.llm.provider == "mock":
        return PolicyIntelligenceExtractor(MockPolicyProvider())
    if settings.llm.provider == "gemini":
        return PolicyIntelligenceExtractor(GeminiPolicyProvider(settings.llm.model))
    else:
        raise RuntimeError(
            f"LLM provider {settings.llm.provider!r} is configured but not implemented. "
            "Use 'mock' or 'gemini'."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--document-type", default="statement")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/processed/llm_policy_intelligence.jsonl")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    settings = load_settings(args.config)
    document_path = settings.data.processed_dir / "fed_documents.csv"
    if not document_path.is_file():
        raise FileNotFoundError(f"Processed documents not found: {document_path}")
    documents = pd.read_csv(document_path)
    if args.document_type:
        if "document_type" not in documents.columns:
            raise ValueError("Document data has no document_type column for filtering.")
        documents = documents.loc[
            documents["document_type"].astype(str).eq(args.document_type)
        ].copy()
    if args.limit is not None:
        documents = documents.head(args.limit)
    destination = Path(args.output).resolve()
    records, reused = run_extraction(
        documents,
        extractor=build_extractor(settings),
        cache_path=destination,
    )
    _write_jsonl_atomic(records, destination)
    counts = pd.Series([record["status"] for record in records]).value_counts().to_dict()
    print(
        json.dumps(
            {
                "rows": len(records),
                "reused_cache_rows": reused,
                "status_counts": counts,
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
