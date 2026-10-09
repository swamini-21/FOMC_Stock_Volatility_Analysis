from fed_policy_intelligence.data.identity import canonicalize_url, document_id, sha256_text


def test_canonicalize_url_removes_tracking_and_fragment() -> None:
    value = "HTTPS://WWW.FederalReserve.gov/a//b/?utm_source=x&z=2#section"

    assert canonicalize_url(value) == "https://www.federalreserve.gov/a/b?z=2"


def test_canonicalize_url_rejects_non_http_urls() -> None:
    assert canonicalize_url("file:///tmp/document.pdf") == ""


def test_document_id_is_stable_for_same_canonical_url() -> None:
    url = "https://www.federalreserve.gov/example"

    assert document_id(url, "first") == document_id(url, "second")


def test_text_hash_changes_with_content() -> None:
    assert sha256_text("one") != sha256_text("two")

