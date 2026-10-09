import io
import json
import logging

import pytest

from fed_policy_intelligence.logging_config import configure_logging


def test_json_logging_includes_context() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    logging.getLogger("test.pipeline").info(
        "validation complete", extra={"record_count": 773}
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "INFO"
    assert payload["message"] == "validation complete"
    assert payload["context"]["record_count"] == 773


def test_invalid_logging_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        configure_logging("verbose")

