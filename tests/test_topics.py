from fed_policy_intelligence.nlp.topics import fit_lda_topics


def test_lda_retains_raw_assignments_and_keywords() -> None:
    documents = [
        "inflation prices policy inflation",
        "inflation price stability policy",
        "employment labor jobs employment",
        "labor market jobs employment",
    ]

    result = fit_lda_topics(documents, topic_count=2, random_state=42, keywords_per_topic=3)

    assert result.method == "sklearn_lda_count_v1"
    assert len(result.assignments) == len(documents)
    assert set(result.topic_keywords) == {0, 1}
    assert all(len(words) == 3 for words in result.topic_keywords.values())
