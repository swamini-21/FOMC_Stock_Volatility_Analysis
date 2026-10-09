"""Reproducible LDA baseline and optional BERTopic comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TopicDiscoveryResult:
    """Raw topic assignments and keywords, without subjective labels."""

    method: str
    assignments: tuple[int, ...]
    topic_keywords: dict[int, tuple[str, ...]]
    model: Any


def fit_lda_topics(
    documents: list[str],
    *,
    topic_count: int,
    random_state: int,
    keywords_per_topic: int = 10,
) -> TopicDiscoveryResult:
    """Fit a count-vector LDA baseline with a deterministic seed."""

    if len(documents) < topic_count or topic_count < 2:
        raise ValueError(
            "LDA requires at least as many documents as topics and at least two topics."
        )
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError("Install topic dependencies with: pip install -e '.[nlp]'") from exc

    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.95)
    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[1] == 0:
        raise ValueError("No vocabulary remained after LDA vectorization.")
    model = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=random_state,
        learning_method="batch",
    )
    document_topics = model.fit_transform(matrix)
    vocabulary = vectorizer.get_feature_names_out()
    keywords = {
        topic_id: tuple(
            vocabulary[index]
            for index in weights.argsort()[-keywords_per_topic:][::-1]
        )
        for topic_id, weights in enumerate(model.components_)
    }
    return TopicDiscoveryResult(
        method="sklearn_lda_count_v1",
        assignments=tuple(int(row.argmax()) for row in document_topics),
        topic_keywords=keywords,
        model={"vectorizer": vectorizer, "estimator": model},
    )


def fit_bertopic_topics(
    documents: list[str], *, random_state: int, min_topic_size: int = 10
) -> TopicDiscoveryResult:
    """Fit the optional embedding-based comparator when BERTopic is installed."""

    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic is optional and not installed; install the 'topic-comparison' extra."
        ) from exc

    # BERTopic does not expose one universal seed parameter. Setting the UMAP
    # component explicitly makes the principal stochastic step reproducible.
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("BERTopic comparison requires umap-learn.") from exc
    umap_model = UMAP(random_state=random_state)
    model = BERTopic(min_topic_size=min_topic_size, umap_model=umap_model, verbose=False)
    assignments, _ = model.fit_transform(documents)
    topic_keywords = {
        int(topic_id): tuple(word for word, _ in (model.get_topic(topic_id) or []))
        for topic_id in sorted(set(assignments))
    }
    return TopicDiscoveryResult(
        method="bertopic_v1",
        assignments=tuple(int(value) for value in assignments),
        topic_keywords=topic_keywords,
        model=model,
    )
