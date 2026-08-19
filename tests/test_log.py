import json
from datetime import UTC, datetime
from types import SimpleNamespace

from app.utils.log import Logger


def _logger():
    log = object.__new__(Logger)
    log.masking_enabled = True
    log.sensitive_keys = set(Logger.DEFAULT_SENSITIVE_KEYS)
    log.max_value_length = 2000
    log.max_collection_items = 30
    log.pretty_json = False
    return log


def _record(message, **extra):
    return {
        "time": datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC),
        "level": SimpleNamespace(name="ERROR"),
        "message": message,
        "extra": extra,
    }


def test_serialize_omits_empty_http_envelope():
    payload = json.loads(
        _logger().serialize(
            _record(
                "CV analysis failed at llm (llm_timeout)",
                request_id="req-1",
                component="analyze",
                event="failed",
                stage="llm",
                error_code="llm_timeout",
                status_code=504,
            )
        )
    )

    assert payload["component"] == "analyze"
    assert payload["stage"] == "llm"
    assert payload["error_code"] == "llm_timeout"
    assert payload["status_code"] == 504
    assert payload["request_id"] == "req-1"
    assert "request" not in payload
    assert "response" not in payload
    assert "user" not in payload
    assert "id" not in payload


def test_serialize_keeps_http_block_and_error_code():
    payload = json.loads(
        _logger().serialize(
            _record(
                "CV analysis failed (llm_timeout)",
                request_id="req-2",
                component="http",
                event="analyze_request",
                error_code="llm_timeout",
                method="POST",
                url="/cv/analyze",
                status_code=504,
                duration_ms=8123.4,
                response_body={"errors": ["llm_timeout"]},
            )
        )
    )

    assert payload["error_code"] == "llm_timeout"
    assert payload["request"]["method"] == "POST"
    assert payload["request"]["url"] == "/cv/analyze"
    assert payload["response"]["status"] == 504
    assert payload["response"]["duration_ms"] == 8123.4
    assert payload["response"]["body"]["errors"] == ["llm_timeout"]


def test_serialize_includes_provider_fields():
    payload = json.loads(
        _logger().serialize(
            _record(
                "LLM call failed (llm_unavailable)",
                component="llm",
                event="call_failed",
                error_code="llm_unavailable",
                status_code=502,
                provider_status=429,
                provider_code="Throttling.RateQuota",
                cause_type="ServiceUnavailableError",
            )
        )
    )

    assert payload["component"] == "llm"
    assert payload["provider_status"] == 429
    assert payload["provider_code"] == "Throttling.RateQuota"
    assert payload["cause_type"] == "ServiceUnavailableError"
