from fed_policy_intelligence.data.preprocessing import clean_document_text


def test_conversational_conference_call_reference_is_preserved() -> None:
    text = "Conference call. We do not support removing this policy discussion. May 6, 2003"

    result = clean_document_text(text)

    assert "We do not support" in result.cleaned_text
    assert "conference_call_heading" not in result.matched_rules


def test_exact_conference_call_heading_is_removed() -> None:
    text = (
        "Conference Call of the Federal Open Market Committee on December 6, 2007\n"
        "The Committee discussed financial conditions."
    )

    result = clean_document_text(text)

    assert result.cleaned_text == "The Committee discussed financial conditions."
    assert "conference_call_heading" in result.matched_rules


def test_policy_terms_and_negation_are_preserved() -> None:
    result = clean_document_text("The Committee will not reduce the policy rate.")

    assert "not" in result.cleaned_text
    assert "policy rate" in result.cleaned_text


def test_empty_value_returns_auditable_empty_result() -> None:
    result = clean_document_text(None)

    assert result.cleaned_text == ""
    assert result.original_length == 0
    assert result.removal_ratio == 0.0

