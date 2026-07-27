import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.errors import AnalysisError
from app.schemas.cv import Source
from app.services.web_scraper import WebScraper


class ScrapeRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2_048)


def authorize(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    settings = get_settings()
    expected = f"Bearer {settings.scraper_worker_token}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_scraper_worker()
    playwright = await async_playwright().start()
    browser = None
    try:
        browser = await playwright.chromium.launch(
            headless=True,
            proxy={"server": settings.scrape_proxy_url},
        )
        app.state.scraper = WebScraper(browser, settings)
        yield
    finally:
        if browser is not None:
            await browser.close()
        await playwright.stop()


app = FastAPI(
    title="CV Analyzer Scraper Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health/live", include_in_schema=False)
async def live():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def ready():
    if not hasattr(app.state, "scraper"):
        raise HTTPException(status_code=503, detail="Not ready")
    proxy = urlsplit(get_settings().scrape_proxy_url)
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(proxy.hostname, proxy.port or 3128),
            timeout=2,
        )
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Egress proxy unavailable") from exc
    return {"status": "ready"}


@app.post("/scrape", response_model=Source, dependencies=[Depends(authorize)])
async def scrape(payload: ScrapeRequest):
    try:
        return await app.state.scraper.fetch(payload.url)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="blocked_or_invalid_url") from exc
    except PlaywrightError as exc:
        raise HTTPException(status_code=502, detail="browser_error") from exc
