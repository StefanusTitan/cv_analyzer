from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen3.7-flash"
    github_access_token: str = ""

    cv_max_files: int = Field(default=3, ge=1, le=10)
    cv_max_file_size_bytes: int = Field(default=10_485_760, ge=1)
    cv_max_total_size_bytes: int = Field(default=20_971_520, ge=1)
    cv_max_pages: int = Field(default=100, ge=1)
    cv_max_extracted_chars: int = Field(default=100_000, ge=1)
    cv_llm_evidence_chars: int = Field(default=50_000, ge=1)
    cv_response_evidence_chars: int = Field(default=20_000, ge=1)
    request_max_size_bytes: int = Field(default=25_000_000, ge=1)
    pdf_extraction_workers: int = Field(default=2, ge=1, le=8)
    pdf_process_budget: int = Field(default=2, ge=1, le=8)
    extraction_concurrency: int = Field(default=2, ge=1, le=8)
    docx_max_entries: int = Field(default=1_000, ge=1)
    docx_max_uncompressed_bytes: int = Field(default=50_000_000, ge=1)
    docx_max_compression_ratio: int = Field(default=100, ge=1)

    scrape_max_links: int = Field(default=10, ge=0, le=50)
    scrape_max_content_chars: int = Field(default=30_000, ge=1)
    scrape_concurrency: int = Field(default=3, ge=1, le=10)
    scrape_timeout_seconds: int = Field(default=15, ge=1, le=120)
    llm_timeout_seconds: int = Field(default=90, ge=1, le=300)
    llm_max_output_tokens: int = Field(default=2_500, ge=256, le=16_000)

    def validate_runtime(self) -> None:
        if not self.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")
        if self.cv_max_total_size_bytes < self.cv_max_file_size_bytes:
            raise RuntimeError(
                "CV_MAX_TOTAL_SIZE_BYTES must be at least CV_MAX_FILE_SIZE_BYTES"
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
