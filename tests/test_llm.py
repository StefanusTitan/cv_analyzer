import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.clients import dashscope as dashscope_client
from app.clients.dashscope import LLMClient


def test_text_completion_uses_native_multimodal_api(monkeypatch):
    call = AsyncMock(
        return_value=SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[{"text": "  assessment  "}])
                    )
                ]
            ),
        )
    )
    monkeypatch.setattr(dashscope_client.AioMultiModalConversation, "call", call)
    client = object.__new__(LLMClient)
    client.api_key = "test-key"
    client.model = "qwen3.7-flash-2026-07-15"
    client.max_output_tokens = 2_500
    client.enable_thinking = False
    client.timeout_seconds = 90

    result = asyncio.run(
        client.text_completion("system prompt", "user prompt", max_tokens=1_200)
    )

    assert result == "assessment"
    call.assert_awaited_once_with(
        api_key="test-key",
        model="qwen3.7-flash-2026-07-15",
        messages=[
            {"role": "system", "content": [{"text": "system prompt"}]},
            {"role": "user", "content": [{"text": "user prompt"}]},
        ],
        temperature=0.3,
        max_tokens=1_200,
        enable_thinking=False,
        request_timeout=90,
    )
