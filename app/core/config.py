from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1"
    dashscope_model: str = "qwen3.7-flash-2026-07-15"
    github_access_token: str = ""
    scraper_worker_url: str = "http://scraper_worker:8010"
    scraper_worker_token: str = ""
    scrape_proxy_url: str = ""
    job_posting_api_url: str = (
        "https://apidev-hrms.duluin.com/api/proxy/v3/employees/job-posting"
    )

    cv_max_files: int = Field(default=3, ge=1, le=10)
    cv_max_file_size_bytes: int = Field(default=10_485_760, ge=1)
    cv_max_total_size_bytes: int = Field(default=20_971_520, ge=1)
    cv_max_pages: int = Field(default=100, ge=1)
    cv_max_extracted_chars: int = Field(default=100_000, ge=1)
    cv_llm_evidence_chars: int = Field(default=50_000, ge=1)
    request_max_size_bytes: int = Field(default=25_000_000, ge=1)
    pdf_extraction_workers: int = Field(default=2, ge=1, le=8)
    pdf_process_budget: int = Field(default=2, ge=1, le=8)
    extraction_concurrency: int = Field(default=2, ge=1, le=8)
    docx_max_entries: int = Field(default=1_000, ge=1)
    docx_max_uncompressed_bytes: int = Field(default=50_000_000, ge=1)
    docx_max_compression_ratio: int = Field(default=100, ge=1)
    job_description_max_chars: int = Field(default=20_000, ge=1, le=100_000)

    scrape_max_links: int = Field(default=6, ge=0, le=50)
    scrape_max_content_chars: int = Field(default=30_000, ge=1)
    scrape_concurrency: int = Field(default=4, ge=1, le=10)
    scrape_timeout_seconds: int = Field(default=8, ge=1, le=120)
    # Wall-clock budget for all external URL enrichment (GitHub + scrape).
    # Partial results are kept and analysis proceeds when the budget elapses.
    enrichment_budget_seconds: float = Field(default=8.0, ge=1.0, le=60.0)
    llm_timeout_seconds: int = Field(default=90, ge=1, le=300)
    llm_max_output_tokens: int = Field(default=2_500, ge=256, le=16_000)
    llm_enable_thinking: bool = False

    def validate_runtime(self) -> None:
        if not self.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")
        if not self.scraper_worker_token:
            raise RuntimeError("SCRAPER_WORKER_TOKEN is required")
        if not self.scraper_worker_url.startswith("http://"):
            raise RuntimeError("SCRAPER_WORKER_URL must be an internal HTTP URL")
        job_posting_api = urlparse(self.job_posting_api_url)
        local_http_api = (
            job_posting_api.scheme == "http"
            and job_posting_api.hostname
            in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
            and job_posting_api.username is None
            and job_posting_api.password is None
        )
        if job_posting_api.scheme != "https" and not local_http_api:
            raise RuntimeError(
                "JOB_POSTING_API_URL must use HTTPS unless a local development "
                "host is used"
            )
        if self.cv_max_total_size_bytes < self.cv_max_file_size_bytes:
            raise RuntimeError(
                "CV_MAX_TOTAL_SIZE_BYTES must be at least CV_MAX_FILE_SIZE_BYTES"
            )

    def validate_scraper_worker(self) -> None:
        if not self.scraper_worker_token:
            raise RuntimeError("SCRAPER_WORKER_TOKEN is required")
        if not self.scrape_proxy_url.startswith("http://"):
            raise RuntimeError("SCRAPE_PROXY_URL is required and must use HTTP")


@lru_cache
def get_settings() -> Settings:
    return Settings()
