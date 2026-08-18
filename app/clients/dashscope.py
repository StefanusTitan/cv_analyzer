from http import HTTPStatus

import dashscope
from dashscope import AioMultiModalConversation
from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    TimeoutException,
    UnsupportedHTTPMethod,
)

from app.core.errors import UpstreamError


class LLMClient:
    def __init__(self, settings):
        dashscope.base_http_api_url = settings.dashscope_base_url
        self.api_key = settings.dashscope_api_key
        self.model = settings.dashscope_model
        self.max_output_tokens = settings.llm_max_output_tokens
        self.enable_thinking = settings.llm_enable_thinking
        self.timeout_seconds = settings.llm_timeout_seconds

    async def close(self) -> None:
        return None

    async def text_completion(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await AioMultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=[
                    {"role": "system", "content": [{"text": system}]},
                    {"role": "user", "content": [{"text": user}]},
                ],
                temperature=0.3,
                max_tokens=max_tokens or self.max_output_tokens,
                enable_thinking=self.enable_thinking,
                request_timeout=self.timeout_seconds,
            )
        except AuthenticationError as exc:
            raise UpstreamError(
                "The LLM provider rejected the configured credentials",
                502,
                "llm_authentication_failed",
            ) from exc
        except ServiceUnavailableError as exc:
            raise UpstreamError(
                "The LLM provider is temporarily unavailable",
                502,
                "llm_unavailable",
            ) from exc
        except (TimeoutException, TimeoutError) as exc:
            raise UpstreamError(
                "The LLM provider timed out", 504, "llm_timeout"
            ) from exc
        except (RequestFailure, InvalidParameter, UnsupportedHTTPMethod) as exc:
            raise UpstreamError(
                "The LLM provider returned an error",
                502,
                "llm_provider_error",
            ) from exc

        if response.status_code != HTTPStatus.OK:
            code = (response.code or "").lower()
            message = response.message or ""
            if response.status_code in {401, 403} or "apikey" in code or "auth" in code:
                raise UpstreamError(
                    "The LLM provider rejected the configured credentials",
                    502,
                    "llm_authentication_failed",
                )
            if response.status_code == 429 or "throttl" in code or "rate" in code:
                raise UpstreamError(
                    "The LLM provider is temporarily unavailable",
                    502,
                    "llm_unavailable",
                )
            if (
                response.status_code == 408
                or "timeout" in code
                or "timeout" in message.lower()
            ):
                raise UpstreamError(
                    "The LLM provider timed out", 504, "llm_timeout"
                )
            raise UpstreamError(
                "The LLM provider returned an error",
                502,
                "llm_provider_error",
            )

        content = self._extract_content(response)
        if not content or not content.strip():
            raise UpstreamError(
                "The LLM provider returned an empty response",
                502,
                "llm_empty_response",
            )
        return content.strip()

    @staticmethod
    def _extract_content(response) -> str | None:
        output = getattr(response, "output", None)
        choices = getattr(output, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None and isinstance(choices[0], dict):
            message = choices[0].get("message")
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        texts = [
            item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            for item in content
        ]
        return "".join(text for text in texts if text)
