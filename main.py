from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
from starlette.exceptions import HTTPException

from app.api.routes.cv import router as cv_router
from app.clients.dashscope import LLMClient
from app.clients.github import GithubClient
from app.core.config import get_settings
from app.exceptions.log import LogError
from app.middlewares.log import LogMiddleware
from app.middlewares.size_limit import RequestSizeLimitMiddleware
from app.services.cv_analyzer import CVAnalyzer
from app.services.web_scraper import WebScraper


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_runtime()
    playwright = await async_playwright().start()
    browser = None
    http_client = httpx.AsyncClient(timeout=15)
    llm = LLMClient(settings)
    try:
        browser = await playwright.chromium.launch(headless=True)
        github = GithubClient(http_client, settings.github_access_token)
        app.state.cv_analyzer = CVAnalyzer(
            settings, llm, github, WebScraper(browser, settings)
        )
        yield
    finally:
        if browser is not None:
            await browser.close()
        await http_client.aclose()
        await llm.close()
        await playwright.stop()


app = FastAPI(lifespan=lifespan)
app.include_router(cv_router)
log = LogError()

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=get_settings().request_max_size_bytes,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LogMiddleware)

app.add_exception_handler(
    RequestValidationError, log.request_validation_exception_handler
)
app.add_exception_handler(HTTPException, log.http_exception_handler)
app.add_exception_handler(Exception, log.unhandled_exception_handler)


@app.get("/")
async def root():
    return {"message": "Server is UP ✅"}
