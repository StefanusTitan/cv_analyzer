import httpx
from pydantic import ValidationError

from app.core.errors import UpstreamError
from app.schemas.cv import Source


class ScraperClient:
    def __init__(self, client: httpx.AsyncClient, settings):
        self.client = client
        self.base_url = settings.scraper_worker_url.rstrip("/")
        self.token = settings.scraper_worker_token
        self.timeout = settings.scrape_timeout_seconds + 10

    async def is_ready(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}/health/ready",
                timeout=3,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def fetch(self, url: str) -> dict:
        try:
            response = await self.client.post(
                f"{self.base_url}/scrape",
                json={"url": url},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
            if response.status_code == 429:
                raise UpstreamError(
                    "The scraping worker is busy", 502, "scraper_worker_busy"
                )
            if response.status_code >= 400:
                raise UpstreamError(
                    "The scraping worker rejected the URL",
                    502,
                    "scraper_worker_rejected",
                )
            return Source.model_validate(response.json()).model_dump()
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "The scraping worker timed out", 504, "scraper_worker_timeout"
            ) from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise UpstreamError(
                "The scraping worker is unavailable", 502, "scraper_worker_unavailable"
            ) from exc
