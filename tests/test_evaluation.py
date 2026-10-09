import pytest

from fed_policy_intelligence.evaluation import evaluate_labels


def test_classification_metrics_and_confusion_matrix() -> None:
    report = evaluate_labels(
        ["hawkish", "hawkish", "dovish", "neutral"],
        ["hawkish", "dovish", "dovish", "neutral"],
        labels=["hawkish", "dovish", "neutral"],
    )

    assert report.sample_size == 4
    assert report.confusion_matrix == ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    assert report.per_class["hawkish"].recall == pytest.approx(0.5)
    assert 0.0 <= report.macro_f1 <= 1.0


def test_metrics_reject_unconfigured_observed_labels() -> None:
    with pytest.raises(ValueError, match="missing"):
        evaluate_labels(["a"], ["b"], labels=["a"])
