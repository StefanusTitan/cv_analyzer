from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    url: str
    type: str
    title: str | None = None
    excerpt: str | None = None


class AnalyzeResult(BaseModel):
    job_posting_id: str
    job_title: str
    analysis: str = Field(min_length=1)
    analysis_en: str = Field(min_length=1)
    sources: list[Source] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    message: str
    result: AnalyzeResult | None
    errors: list[str] | None = None
