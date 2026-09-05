"""Endpoint-level contract tests for ``POST /cv/analyze``.

These tests pin the worker-facing contract used by the durable
``service_employees`` worker:

* success shape contains one requested language and ``result.analysis``
* permanent failures return a 4xx with a stable machine-readable ``errors`` code
* retryable/upstream failures return a 5xx with a stable code
* responses never leak exception stacks, upstream bodies, or extracted CV text
* concurrent requests do not mix content or results
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI
from office_oxide import create_from_markdown
from pdf_oxide import Pdf

from app.api.routes.cv import router as cv_router
from app.clients.hrms import JobPostingClient
from app.core.errors import AnalysisError, UpstreamError
from app.middlewares.size_limit import RequestSizeLimitMiddleware
from app.services.cv_analyzer import CVAnalyzer

JOB_POSTING_ID = "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff"
OTHER_JOB_POSTING_ID = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "cv_max_file_size_bytes": 10_485_760,
        "cv_max_total_size_bytes": 20_971_520,
        "cv_max_pages": 100,
        "cv_max_extracted_chars": 100_000,
        "cv_llm_evidence_chars": 50_000,
        "pdf_extraction_workers": 2,
        "pdf_process_budget": 2,
        "extraction_concurrency": 2,
        "docx_max_entries": 1_000,
        "docx_max_uncompressed_bytes": 50_000_000,
        "docx_max_compression_ratio": 100,
        "scrape_max_links": 0,
        "scrape_concurrency": 2,
        "enrichment_budget_seconds": 8.0,
        "llm_max_output_tokens": 8_000,
        "request_max_size_bytes": 25_000_000,
        "job_posting_api_url": "https://gateway.test/job-posting",
        "job_description_max_chars": 20_000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeLLM:
    """Echoes the request's job_posting_id so concurrent requests can be told apart."""

    def __init__(self, response: str | None = None, error: UpstreamError | None = None):
        self.response = response
        self.error = error
        self.calls = 0

    async def text_completion(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        match = re.match(r"JOB POSTING ID: ([^\n]+)", user)
        jid = match.group(1) if match else "unknown"
        if "bahasa Inggris" in system:
            return f"The candidate is a moderate fit for {jid} based on [[S1]]."
        return f"Kandidat cukup sesuai untuk {jid} berdasarkan [[S1]]."


class FakeJobPostingClient:
    def __init__(self, posting: Any | None = None, error: UpstreamError | None = None):
        self.posting = posting
        self.error = error

    async def fetch(self, job_posting_id: str):
        if self.error is not None:
            raise self.error
        if self.posting is not None:
            return self.posting
        # Mirror the real client: echo the requested id so concurrent requests
        # can be distinguished by their persisted analysis.
        return SimpleNamespace(
            id=job_posting_id,
            title="Backend Engineer",
            description="Build reliable Python APIs.",
            other_job_titles=("Python Developer",),
        )


class NoopEnricher:
    async def fetch(self, url: str):
        raise AssertionError(f"Unexpected enrichment call: {url}")


def default_posting(job_posting_id: str = JOB_POSTING_ID) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_posting_id,
        title="Backend Engineer",
        description="Build reliable Python APIs.",
        other_job_titles=("Python Developer",),
    )


def build_app(
    analyzer: CVAnalyzer,
    *,
    max_bytes: int = 25_000_000,
) -> FastAPI:
    app = FastAPI()
    app.include_router(cv_router)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)
    app.state.cv_analyzer = analyzer
    return app


def make_analyzer(
    *,
    settings: SimpleNamespace | None = None,
    llm: FakeLLM | None = None,
    job_postings: Any | None = None,
    github: Any | None = None,
    gitlab: Any | None = None,
    bitbucket: Any | None = None,
    huggingface: Any | None = None,
    oembed: Any | None = None,
    scraper: Any | None = None,
) -> CVAnalyzer:
    return CVAnalyzer(
        settings or make_settings(),
        llm or FakeLLM(),
        github or NoopEnricher(),
        gitlab or NoopEnricher(),
        bitbucket or NoopEnricher(),
        huggingface or NoopEnricher(),
        oembed or NoopEnricher(),
        scraper or NoopEnricher(),
        # By default the fake echoes the requested job_posting_id; tests that need
        # a fixed posting pass one explicitly.
        job_postings or FakeJobPostingClient(),
    )


def make_pdf(text: str = "Candidate built a production Python API.") -> bytes:
    return Pdf.from_text(text).to_bytes()


def make_docx(text: str = "Candidate built a production Python API.") -> bytes:
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "cv.docx")
    create_from_markdown(text, "docx", path)
    with open(path, "rb") as handle:
        return handle.read()


def run(coro):
    return asyncio.run(coro)


async def post(
    app: FastAPI,
    *,
    job_posting_id: str = JOB_POSTING_ID,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    links: list[str] | None = None,
    file_titles: list[str] | None = None,
    omit_files: bool = False,
    language: str = "id",
) -> httpx.Response:
    data: dict[str, Any] = {
        "job_posting_id": job_posting_id,
        "language": language,
    }
    if links is not None:
        data["links"] = links
    if file_titles is not None:
        data["file_titles"] = file_titles
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        kwargs: dict[str, Any] = {"data": data}
        if not omit_files:
            kwargs["files"] = files or [
                ("files", ("cv.pdf", make_pdf(), "application/pdf"))
            ]
        return await client.post("/cv/analyze", **kwargs)


# ---------------------------------------------------------------------------
# Success contract
# ---------------------------------------------------------------------------


def test_valid_pdf_returns_200_and_nonempty_analysis():
    app = build_app(make_analyzer())
    response = run(
        post(app, files=[("files", ("cv.pdf", make_pdf(), "application/pdf"))])
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "CV analyzed successfully"
    assert body["result"] is not None
    assert body["result"]["job_posting_id"] == JOB_POSTING_ID
    assert body["result"]["job_title"] == "Backend Engineer"
    assert body["result"]["language"] == "id"
    assert body["result"]["analysis"]
    assert "analysis_en" not in body["result"]
    assert isinstance(body["result"]["sources"], list)
    assert isinstance(body["result"]["warnings"], list)
    assert "<b>Sumber:</b>" in body["result"]["analysis"]
    assert "<li>[1] cv.pdf</li>" in body["result"]["analysis"]


def test_file_titles_appear_in_source_legend():
    app = build_app(
        make_analyzer(
            llm=FakeLLM(
                response="Kandidat cukup sesuai [[S1]]."
            )
        )
    )
    response = run(
        post(
            app,
            files=[("files", ("uuid.pdf", make_pdf(), "application/pdf"))],
            file_titles=["Kirimkan CVmu"],
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert "[1]" in body["result"]["analysis"]
    assert "<li>[1] Kirimkan CVmu</li>" in body["result"]["analysis"]
    assert body["result"]["sources"][0]["title"] == "Kirimkan CVmu"


def test_valid_docx_returns_200_and_nonempty_analysis():
    app = build_app(make_analyzer())
    response = run(
        post(
            app,
            files=[
                (
                    "files",
                    (
                        "cv.docx",
                        make_docx(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["analysis"]


def test_multiple_accepted_files_still_work():
    app = build_app(make_analyzer())
    response = run(
        post(
            app,
            files=[
                (
                    "files",
                    ("cv1.pdf", make_pdf("Python API builder"), "application/pdf"),
                ),
                (
                    "files",
                    (
                        "cv2.docx",
                        make_docx("Led a team of engineers"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                ),
            ],
        )
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result["sources"]) == 2
    assert result["analysis"]


def test_selected_portfolio_link_is_enriched():
    class RecordingScraper:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return {
                "id": "web:portfolio",
                "url": url,
                "type": "website",
                "title": "Portfolio",
                "excerpt": "Production application portfolio",
            }

    scraper = RecordingScraper()
    app = build_app(
        make_analyzer(
            settings=make_settings(scrape_max_links=1),
            scraper=scraper,
        )
    )
    portfolio_url = "https://portfolio.example.test/candidate"
    response = run(post(app, links=[portfolio_url], omit_files=True))

    assert response.status_code == 200
    assert scraper.urls == [portfolio_url]
    assert "web:portfolio" in {
        source["id"] for source in response.json()["result"]["sources"]
    }


def test_success_response_preserves_result_analysis_field():
    """The employee worker persists the requested ``result.analysis``."""
    app = build_app(
        make_analyzer(
            llm=FakeLLM(
                response="Display-ready hiring summary. [[S1]]"
            )
        )
    )
    response = run(post(app, language="en"))
    assert response.status_code == 200
    assert response.json()["result"]["language"] == "en"
    assert response.json()["result"]["analysis"].startswith(
        "Display-ready hiring summary."
    )
    assert "<b>Sources:</b>" in response.json()["result"]["analysis"]
    assert "analysis_en" not in response.json()["result"]


def test_analysis_output_strips_internal_document_markers():
    app = build_app(
        make_analyzer(
            llm=FakeLLM(
                response="Cukup sesuai [document:0]; verifikasi backend [job_description] "
                "dan portofolio [github:owner/repo] and [web:https://x.test/]."
            )
        )
    )
    response = run(post(app))
    analysis = response.json()["result"]["analysis"]
    assert "[document:" not in analysis
    assert "[github:" not in analysis
    assert "[web:" not in analysis
    assert "[job_description]" not in analysis
    assert "[cv_and_resume]" not in analysis


# ---------------------------------------------------------------------------
# Employee-worker contract test
# ---------------------------------------------------------------------------


def test_employee_worker_contract_persists_result_analysis():
    """Exactly the interaction the durable employee-service worker performs."""
    analyzer = make_analyzer(
        llm=FakeLLM(
            response="Strong fit with verifiable backend experience. [[S1]]"
        ),
        job_postings=FakeJobPostingClient(default_posting()),
    )
    app = build_app(analyzer)
    response = run(
        post(
            app,
            job_posting_id=JOB_POSTING_ID,
            files=[("files", ("applicant.pdf", make_pdf(), "application/pdf"))],
            language="en",
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] is not None
    assert body["result"]["language"] == "en"
    analysis = body["result"]["analysis"]
    assert isinstance(analysis, str)
    assert analysis.startswith("Strong fit with verifiable backend experience.")
    assert "<b>Sources:</b>" in analysis
    assert "analysis_en" not in body["result"]


# ---------------------------------------------------------------------------
# Permanent / input failure codes
# ---------------------------------------------------------------------------


def test_missing_files_returns_missing_files():
    app = build_app(make_analyzer())
    response = run(post(app, omit_files=True))
    assert response.status_code == 400
    body = response.json()
    assert body["result"] is None
    assert body["errors"] == ["missing_files"]


def test_invalid_language_returns_invalid_language():
    app = build_app(make_analyzer())
    response = run(post(app, language="fr"))
    assert response.status_code == 400
    assert response.json()["errors"] == ["invalid_language"]


def test_oversized_file_returns_file_size_exceeded():
    settings = make_settings(cv_max_file_size_bytes=50)
    app = build_app(make_analyzer(settings=settings))
    response = run(
        post(app, files=[("files", ("cv.pdf", make_pdf(), "application/pdf"))])
    )
    assert response.status_code == 413
    assert response.json()["errors"] == ["file_size_exceeded"]


def test_oversized_combined_input_returns_total_size_exceeded():
    settings = make_settings(
        cv_max_file_size_bytes=10_000_000, cv_max_total_size_bytes=100
    )
    app = build_app(make_analyzer(settings=settings))
    files = [
        ("files", ("cv1.pdf", make_pdf("first document"), "application/pdf")),
        ("files", ("cv2.pdf", make_pdf("second document"), "application/pdf")),
    ]
    response = run(post(app, files=files))
    assert response.status_code == 413
    assert response.json()["errors"] == ["total_size_exceeded"]


def test_unsupported_type_returns_415_unsupported_format():
    app = build_app(make_analyzer())
    response = run(
        post(
            app,
            files=[("files", ("cv.txt", b"plain text resume", "text/plain"))],
        )
    )
    assert response.status_code == 415
    assert response.json()["errors"] == ["unsupported_format"]


def test_invalid_pdf_returns_invalid_document():
    app = build_app(make_analyzer())
    invalid = b"%PDF-1.7\n%this is not a real pdf body\n"
    response = run(post(app, files=[("files", ("cv.pdf", invalid, "application/pdf"))]))
    assert response.status_code == 422
    assert response.json()["errors"] == ["invalid_document"]


def test_invalid_docx_returns_invalid_document():
    app = build_app(make_analyzer())
    response = run(
        post(
            app,
            files=[
                (
                    "files",
                    (
                        "cv.docx",
                        b"not a zip archive",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
        )
    )
    assert response.status_code == 422
    assert response.json()["errors"] == ["invalid_document"]


def test_document_with_no_usable_text_returns_empty_document():
    app = build_app(make_analyzer())
    response = run(
        post(
            app,
            files=[
                (
                    "files",
                    (
                        "cv.docx",
                        make_docx(""),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
        )
    )
    assert response.status_code == 422
    assert response.json()["errors"] == ["empty_document"]


def test_invalid_job_posting_uuid_returns_invalid_job_posting_id():
    # Use the real JobPostingClient so UUID validation produces the stable code.
    settings = make_settings()
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    job_postings = JobPostingClient(httpx.AsyncClient(transport=transport), settings)
    app = build_app(make_analyzer(settings=settings, job_postings=job_postings))
    response = run(post(app, job_posting_id="not-a-uuid"))
    assert response.status_code == 400
    assert response.json()["errors"] == ["invalid_job_posting_id"]


def test_missing_job_posting_returns_permanent_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    settings = make_settings()
    transport = httpx.MockTransport(handler)
    job_postings = JobPostingClient(httpx.AsyncClient(transport=transport), settings)
    analyzer = make_analyzer(settings=settings, job_postings=job_postings)
    # Provide a syntactically valid UUID so validation passes; HRMS returns 404.
    analyzer.job_postings = job_postings
    app = build_app(analyzer)
    response = run(
        post(
            app,
            job_posting_id=OTHER_JOB_POSTING_ID,
            files=[("files", ("cv.pdf", make_pdf(), "application/pdf"))],
        )
    )
    assert response.status_code == 404
    assert response.json()["errors"] == ["job_posting_not_found"]


# ---------------------------------------------------------------------------
# Retryable / upstream failure codes (5xx)
# ---------------------------------------------------------------------------


def _hrms_app(handler) -> FastAPI:
    settings = make_settings()
    transport = httpx.MockTransport(handler)
    job_postings = JobPostingClient(httpx.AsyncClient(transport=transport), settings)
    return build_app(make_analyzer(settings=settings, job_postings=job_postings))


def test_hrms_timeout_returns_retryable_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    response = run(post(_hrms_app(handler)))
    assert response.status_code == 504
    assert response.json()["errors"] == ["job_posting_timeout"]


def test_hrms_unavailable_returns_retryable_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    response = run(post(_hrms_app(handler)))
    assert response.status_code == 502
    assert response.json()["errors"] == ["job_posting_unavailable"]


def test_llm_timeout_returns_retryable_5xx():
    analyzer = make_analyzer(
        llm=FakeLLM(error=UpstreamError("llm timed out", 504, "llm_timeout")),
    )
    app = build_app(analyzer)
    response = run(post(app))
    assert response.status_code == 504
    assert response.json()["errors"] == ["llm_timeout"]


def test_llm_unavailable_returns_retryable_5xx():
    analyzer = make_analyzer(
        llm=FakeLLM(error=UpstreamError("llm unavailable", 502, "llm_unavailable")),
    )
    app = build_app(analyzer)
    response = run(post(app))
    assert response.status_code == 502
    assert response.json()["errors"] == ["llm_unavailable"]


# ---------------------------------------------------------------------------
# Cleanup and safety
# ---------------------------------------------------------------------------


def test_uploads_close_on_success():
    analyzer = make_analyzer()
    upload = __import__("fastapi").UploadFile(
        filename="cv.pdf", file=io.BytesIO(make_pdf())
    )
    run(analyzer.analyze(JOB_POSTING_ID, [upload]))
    assert upload.file.closed


def test_uploads_close_on_permanent_failure():
    analyzer = make_analyzer()
    # A zero-text DOCX yields empty_document; uploads must still close.
    upload = __import__("fastapi").UploadFile(
        filename="cv.docx", file=io.BytesIO(make_docx(""))
    )
    try:
        run(analyzer.analyze(JOB_POSTING_ID, [upload]))
    except AnalysisError:
        pass
    assert upload.file.closed


def test_concurrent_requests_do_not_mix_results():
    analyzer = make_analyzer()  # shared analyzer instance, like app.state
    app = build_app(analyzer)

    async def call(job_posting_id: str) -> httpx.Response:
        return await post(
            app,
            job_posting_id=job_posting_id,
            files=[
                (
                    "files",
                    (
                        "cv.pdf",
                        make_pdf(f"resume for {job_posting_id}"),
                        "application/pdf",
                    ),
                )
            ],
        )

    async def main():
        resp_a, resp_b = await asyncio.gather(
            call(JOB_POSTING_ID), call(OTHER_JOB_POSTING_ID)
        )
        return resp_a, resp_b

    resp_a, resp_b = run(main())
    analysis_a = resp_a.json()["result"]["analysis"]
    analysis_b = resp_b.json()["result"]["analysis"]
    assert JOB_POSTING_ID in analysis_a
    assert OTHER_JOB_POSTING_ID in analysis_b
    assert OTHER_JOB_POSTING_ID not in analysis_a
    assert JOB_POSTING_ID not in analysis_b


def test_responses_never_expose_exception_stacks_or_cv_text():
    marker = "UNIQUE_CV_MARKER_DO_NOT_LEAK_12345"
    analyzer = make_analyzer(
        llm=FakeLLM(error=UpstreamError("llm unavailable", 502, "llm_unavailable")),
    )
    app = build_app(analyzer)
    response = run(
        post(
            app,
            files=[("files", ("cv.pdf", make_pdf(marker), "application/pdf"))],
        )
    )
    body_text = response.text
    assert response.status_code == 502
    assert "Traceback" not in body_text
    assert marker not in body_text
    body = response.json()
    assert set(body) <= {"message", "result", "errors"}
    assert body["result"] is None
    assert body["errors"] == ["llm_unavailable"]
