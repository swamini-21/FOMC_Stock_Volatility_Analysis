from fed_policy_intelligence.nlp.policy import PolicyStance, classify_policy_stance


def test_policy_baseline_identifies_hawkish_evidence() -> None:
    result = classify_policy_stance(
        "The Committee raised the target range. Inflation risks remain elevated."
    )

    assert result.stance == PolicyStance.HAWKISH
    assert result.hawkish_matches
    assert result.supporting_evidence


def test_policy_baseline_keeps_mixed_evidence() -> None:
    result = classify_policy_stance(
        "The stance remains restrictive policy. Downside risks to employment have increased."
    )

    assert result.stance == PolicyStance.NEUTRAL_OR_MIXED
    assert result.hawkish_matches and result.dovish_matches


def test_policy_baseline_abstains_without_explicit_evidence() -> None:
    result = classify_policy_stance("The Committee reviewed developments in financial markets.")

    assert result.stance == PolicyStance.INSUFFICIENT_EVIDENCE
    assert result.supporting_evidence == ()


def test_policy_baseline_does_not_count_nearby_negated_phrase() -> None:
    result = classify_policy_stance("The Committee did not raise the target range.")

    assert result.stance == PolicyStance.INSUFFICIENT_EVIDENCE
