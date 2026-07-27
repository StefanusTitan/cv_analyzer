from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.core.errors import UpstreamError


class LLMClient:
    def __init__(self, settings):
        self.client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        self.model = settings.dashscope_model
        self.max_output_tokens = settings.llm_max_output_tokens
        self.enable_thinking = settings.llm_enable_thinking

    async def close(self) -> None:
        await self.client.close()

    async def text_completion(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=max_tokens or self.max_output_tokens,
                extra_body={"enable_thinking": self.enable_thinking},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except AuthenticationError as exc:
            raise UpstreamError(
                "The LLM provider rejected the configured credentials",
                502,
                "llm_authentication_failed",
            ) from exc
        except APITimeoutError as exc:
            raise UpstreamError(
                "The LLM provider timed out", 504, "llm_timeout"
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise UpstreamError(
                "The LLM provider is temporarily unavailable",
                502,
                "llm_unavailable",
            ) from exc
        except APIStatusError as exc:
            raise UpstreamError(
                "The LLM provider returned an error",
                502,
                "llm_provider_error",
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise UpstreamError(
                "The LLM provider returned an empty response",
                502,
                "llm_empty_response",
            )
        return content.strip()
