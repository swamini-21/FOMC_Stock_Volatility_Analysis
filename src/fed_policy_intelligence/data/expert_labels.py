"""Load and validate the external expert-labelled FOMC stance dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

DATASET_ID = "gtfintechlab/fomc_communication"
DATASET_REVISION = "6b0283f55f0005a6d38d49f271d795c21fccc1a3"
LABEL_NAMES = {0: "dovish", 1: "hawkish", 2: "neutral"}
REQUIRED_COLUMNS = ("sentence", "label")


@dataclass(frozen=True)
class ExpertDataset:
    """Validated train/test frames plus immutable source metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    dataset_id: str
    revision: str


def normalize_expert_split(rows: Iterable[Mapping[str, Any]], *, split: str) -> pd.DataFrame:
    """Normalize one Hub split and reject undocumented schema or labels."""

    frame = pd.DataFrame.from_records(rows)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Expert dataset split '{split}' is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"Expert dataset split '{split}' is empty.")

    frame = frame.copy()
    frame["sentence"] = frame["sentence"].astype(str).str.strip()
    if frame["sentence"].eq("").any():
        raise ValueError(f"Expert dataset split '{split}' contains empty sentences.")

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin(LABEL_NAMES).all():
        observed = sorted(frame["label"].astype(str).unique().tolist())
        raise ValueError(
            f"Expert dataset split '{split}' has unsupported labels: {observed}"
        )
    frame["label_id"] = numeric_labels.astype(int)
    frame["policy_stance"] = frame["label_id"].map(LABEL_NAMES)
    frame["split"] = split
    return frame.drop(columns=["label"]).reset_index(drop=True)


def load_expert_dataset(
    *,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    cache_dir: str | None = None,
) -> ExpertDataset:
    """Load the public dataset through Hugging Face and validate its contract."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets dependency is missing. "
            "Install the project with: pip install -e '.[nlp]'"
        ) from exc

    loaded = load_dataset(dataset_id, revision=revision, cache_dir=cache_dir)
    missing_splits = sorted({"train", "test"} - set(loaded))
    if missing_splits:
        raise ValueError(f"Expert dataset is missing required splits: {missing_splits}")
    return ExpertDataset(
        train=normalize_expert_split(loaded["train"], split="train"),
        test=normalize_expert_split(loaded["test"], split="test"),
        dataset_id=dataset_id,
        revision=revision,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate remote availability and print a compact, non-mutating summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    dataset = load_expert_dataset(
        dataset_id=args.dataset_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    summary = {
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "train_rows": len(dataset.train),
        "test_rows": len(dataset.test),
        "train_labels": dataset.train["policy_stance"].value_counts().to_dict(),
        "test_labels": dataset.test["policy_stance"].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
