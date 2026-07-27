import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app.clients.scraper import ScraperClient
from app.core.errors import UpstreamError
from app.scraper import worker


def settings():
    return SimpleNamespace(
        scraper_worker_url="http://scraper_worker:8010",
        scraper_worker_token="secret-token",
        scrape_timeout_seconds=15,
    )


def test_worker_service_token(monkeypatch):
    monkeypatch.setattr(
        worker,
        "get_settings",
        lambda: SimpleNamespace(scraper_worker_token="secret-token"),
    )
    worker.authorize("Bearer secret-token")
    with pytest.raises(HTTPException) as error:
        worker.authorize("Bearer wrong-token")
    assert error.value.status_code == 401


def test_scraper_client_authenticates_and_validates_source():
    async def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer secret-token"
        return httpx.Response(
            200,
            json={
                "id": "web:1234",
                "url": "https://example.com/",
                "type": "website",
                "title": "Example",
                "excerpt": "Evidence",
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ScraperClient(client, settings()).fetch("https://example.com")

    assert asyncio.run(run())["id"] == "web:1234"


def test_scraper_client_maps_worker_failure():
    async def handler(request: httpx.Request):
        return httpx.Response(422, json={"detail": "blocked_or_invalid_url"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await ScraperClient(client, settings()).fetch("https://example.com")

    with pytest.raises(UpstreamError) as error:
        asyncio.run(run())
    assert error.value.code == "scraper_worker_rejected"
