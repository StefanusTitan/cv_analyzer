from dataclasses import dataclass
from uuid import UUID

import httpx

from app.core.errors import UpstreamError
from app.utils.html import html_to_text


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
            raise UpstreamError(
                "The job posting ID is invalid", 400, "invalid_job_posting_id"
            ) from exc

        try:
            response = await self.client.get(
                f"{self.base_url}/{normalized_id}"
            )
            if response.status_code == 404:
                raise UpstreamError(
                    "The job posting was not found", 404, "job_posting_not_found"
                )
            response.raise_for_status()
            payload = response.json()
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "The job posting service timed out", 504, "job_posting_timeout"
            ) from exc
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise UpstreamError(
                "The job posting service is unavailable",
                502,
                "job_posting_unavailable",
            ) from exc

        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise UpstreamError(
                "The job posting service returned an invalid response",
                502,
                "job_posting_invalid_response",
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise UpstreamError(
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
            raise UpstreamError(
                "The job posting service returned an invalid response",
                502,
                "job_posting_invalid_response",
            )
        return JobPosting(
            id=normalized_id,
            title=" ".join(title.split())[:500],
            description=html_to_text(description, self.description_max_chars),
        )
