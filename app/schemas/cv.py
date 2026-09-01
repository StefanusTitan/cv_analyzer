from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    url: str
    type: str
    kind: str | None = None
    access_status: str | None = None
    title: str | None = None
    excerpt: str | None = None


class AnalyzeResult(BaseModel):
    job_posting_id: str
    job_title: str
    language: Literal["id", "en"]
    analysis: str = Field(min_length=1)
    sources: list[Source] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    message: str
    result: AnalyzeResult | None
    errors: list[str] | None = None
