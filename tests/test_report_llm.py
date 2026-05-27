from __future__ import annotations

from baseball_pose.pipeline.report_llm import (
    _build_chat_request,
    _extract_report_text,
    _normalize_base_url,
)


def test_build_chat_request_includes_model_temperature_and_messages() -> None:
    request_body = _build_chat_request("hello world", model="gpt-test", temperature=0.3)

    assert request_body["model"] == "gpt-test"
    assert request_body["temperature"] == 0.3
    assert request_body["messages"][0]["role"] == "system"
    assert request_body["messages"][1]["content"] == "hello world"


def test_extract_report_text_reads_standard_chat_completion_shape() -> None:
    response_body = {
        "choices": [
            {
                "message": {
                    "content": "This is the final report.",
                }
            }
        ]
    }

    assert _extract_report_text(response_body) == "This is the final report."


def test_normalize_base_url_accepts_root_or_v1() -> None:
    assert _normalize_base_url("https://example.com") == "https://example.com/v1"
    assert _normalize_base_url("https://example.com/") == "https://example.com/v1"
    assert _normalize_base_url("https://example.com/v1") == "https://example.com/v1"
