from dataclasses import dataclass
from uuid import UUID

import httpx

from app.core.errors import UpstreamError
from app.utils.html import html_to_text
from app.utils.log import logger


@dataclass(frozen=True)
class JobPosting:
    id: str
    title: str
    description: str


class JobPostingClient:
    def __init__(self, client: httpx.AsyncClient, settings):
        self.client = client
        self.base_url = settings.job_posting_api_url.rstrip("/")
        self.description_max_chars = settings.job_description_max_chars

    async def fetch(self, job_posting_id: str) -> JobPosting:
        try:
            normalized_id = str(UUID(job_posting_id))
        except ValueError as exc:
            self._fail(
                "The job posting ID is invalid",
                400,
                "invalid_job_posting_id",
                cause=exc,
            )

        try:
            response = await self.client.get(
                f"{self.base_url}/{normalized_id}"
            )
            if response.status_code == 404:
                self._fail(
                    "The job posting was not found",
                    404,
                    "job_posting_not_found",
                    http_status=404,
                )
            response.raise_for_status()
            payload = response.json()
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            self._fail(
                "The job posting service timed out",
                504,
                "job_posting_timeout",
                cause=exc,
            )
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            self._fail(
                "The job posting service is unavailable",
                502,
                "job_posting_unavailable",
                cause=exc,
            )

        if not isinstance(payload, dict) or payload.get("success") is not True:
            self._fail(
                "The job posting service returned an invalid response",
                502,
                "job_posting_invalid_response",
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            self._fail(
                "The job posting service returned an invalid response",
                502,
                "job_posting_invalid_response",
            )
        title = data.get("title")
        description = data.get("description")
        if (
            str(data.get("id")) != normalized_id
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(description, str)
        ):
            self._fail(
                "The job posting service returned an invalid response",
                502,
                "job_posting_invalid_response",
            )
        return JobPosting(
            id=normalized_id,
            title=" ".join(title.split())[:500],
            description=html_to_text(description, self.description_max_chars),
        )

    def _fail(
        self,
        message: str,
        status_code: int,
        error_code: str,
        *,
        cause: BaseException | None = None,
        http_status: int | None = None,
    ) -> None:
        logger.bind(
            component="job_posting",
            event="fetch_failed",
            error_code=error_code,
            status_code=status_code,
            http_status=http_status,
            cause_type=type(cause).__name__ if cause else None,
        ).error(f"Job posting fetch failed ({error_code})")
        error = UpstreamError(message, status_code, error_code)
        if cause is not None:
            raise error from cause
        raise error
