from dataclasses import dataclass
from typing import NoReturn
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
    other_job_titles: tuple[str, ...]


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

        payload = await self._get_payload(f"{self.base_url}/{normalized_id}")
        data = payload.get("data")
        if not isinstance(data, dict):
            self._invalid_response()
        title = data.get("title")
        description = data.get("description")
        company_id = data.get("company_id")
        if (
            str(data.get("id")) != normalized_id
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(description, str)
            or not isinstance(company_id, str)
            or not company_id.strip()
        ):
            self._invalid_response()
        title = " ".join(title.split())[:500]
        other_job_titles = await self._fetch_titles(company_id, normalized_id, title)
        return JobPosting(
            id=normalized_id,
            title=title,
            description=html_to_text(description, self.description_max_chars),
            other_job_titles=other_job_titles,
        )

    async def _fetch_titles(
        self, company_id: str, posting_id: str, current_title: str
    ) -> tuple[str, ...]:
        titles: dict[str, str] = {}
        page = 1
        while True:
            payload = await self._get_payload(
                self.base_url,
                params={
                    "company_id": company_id,
                    "status": "active",
                    "publish_status": "published",
                    "page": str(page),
                    "limit": "200",
                    "sort": "id",
                    "order": "ASC",
                },
            )
            rows = payload.get("data")
            meta = payload.get("meta")
            if (
                not isinstance(rows, list)
                or not isinstance(meta, dict)
                or meta.get("currentPage") != page
                or not isinstance(meta.get("hasNextPage"), bool)
                or (meta["hasNextPage"] and not rows)
            ):
                self._invalid_response()
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or row.get("company_id") != company_id
                    or row.get("status") != "active"
                    or row.get("publish_status") != "published"
                    or not isinstance(row.get("title"), str)
                    or not row["title"].strip()
                ):
                    self._invalid_response()
                title = " ".join(row["title"].split())[:500]
                if row.get("id") != posting_id and title.casefold() != current_title.casefold():
                    titles.setdefault(title.casefold(), title)
            if not meta["hasNextPage"]:
                return tuple(titles.values())
            page += 1

    def _invalid_response(self) -> NoReturn:
        self._fail(
            "The job posting service returned an invalid response",
            502,
            "job_posting_invalid_response",
        )

    async def _get_payload(self, url: str, *, params: dict[str, str] | None = None) -> dict:
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 404 and url != self.base_url:
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
        return payload

    def _fail(
        self,
        message: str,
        status_code: int,
        error_code: str,
        *,
        cause: BaseException | None = None,
        http_status: int | None = None,
    ) -> NoReturn:
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
