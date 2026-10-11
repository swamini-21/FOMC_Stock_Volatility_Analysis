import pytest

from fed_policy_intelligence.data.expert_labels import normalize_expert_split


def test_expert_labels_are_normalized_to_policy_names() -> None:
    rows = [
        {"sentence": "Policy accommodation may be reduced.", "label": 1, "year": 2022},
        {"sentence": "Rates may remain low.", "label": 0, "year": 2020},
        {"sentence": "The Committee reviewed developments.", "label": 2, "year": 2019},
    ]

    result = normalize_expert_split(rows, split="train")

    assert result["policy_stance"].tolist() == ["hawkish", "dovish", "neutral"]
    assert result["label_id"].tolist() == [1, 0, 2]
    assert result["split"].eq("train").all()
    assert "year" in result.columns


def test_expert_dataset_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unsupported labels"):
        normalize_expert_split([{"sentence": "Text", "label": 9}], split="test")


def test_expert_dataset_rejects_empty_sentences() -> None:
    with pytest.raises(ValueError, match="empty sentences"):
        normalize_expert_split([{"sentence": "  ", "label": 2}], split="test")
