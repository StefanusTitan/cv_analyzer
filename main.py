from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from app.api.routes.cv import router as cv_router
from app.clients.dashscope import LLMClient
from app.clients.github import GithubClient
from app.clients.hrms import JobPostingClient
from app.clients.scraper import ScraperClient
from app.core.config import get_settings
from app.exceptions.log import LogError
from app.middlewares.log import LogMiddleware
from app.middlewares.size_limit import RequestSizeLimitMiddleware
from app.services.cv_analyzer import CVAnalyzer

CORS_ALLOWED_ORIGINS = ["https://workin-dev.duluin.id"]
CORS_ALLOW_ORIGIN_REGEX = r"https://[a-z0-9-]+\.workin\.duluin\.(com|id)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_runtime()
    http_client = httpx.AsyncClient(timeout=15)
    llm = LLMClient(settings)
    try:
        github = GithubClient(http_client, settings.github_access_token)
        job_postings = JobPostingClient(http_client, settings)
        scraper = ScraperClient(http_client, settings)
        app.state.scraper_client = scraper
        app.state.cv_analyzer = CVAnalyzer(settings, llm, github, scraper, job_postings)
        yield
    finally:
        await http_client.aclose()
        await llm.close()


app = FastAPI(lifespan=lifespan)
app.include_router(cv_router)
log = LogError()
startup_settings = get_settings()

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=startup_settings.request_max_size_bytes,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Forwarded-Host",
        "X-Request-ID",
    ],
    expose_headers=["X-Request-ID"],
    max_age=600,
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


@app.get("/health/live", include_in_schema=False)
async def live():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def ready():
    if (
        not hasattr(app.state, "cv_analyzer")
        or not await app.state.scraper_client.is_ready()
    ):
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ready"}
