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


class Rating(BaseModel):
    score: int = Field(ge=0, le=100)
    scale: int = 100


class TokenEstimation(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AnalyzeResult(BaseModel):
    job_posting_id: str
    job_title: str
    language: Literal["id", "en"]
    analysis: str = Field(min_length=1)
    rating: Rating | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    tokens: TokenEstimation | None = None
    sources: list[Source] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)



class AnalyzeResponse(BaseModel):
    message: str
    result: AnalyzeResult | None
    errors: list[str] | None = None

