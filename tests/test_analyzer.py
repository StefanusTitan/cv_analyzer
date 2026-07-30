import asyncio
import io
import json
from types import SimpleNamespace

import pymupdf
from fastapi import UploadFile

from app.services.cv_analyzer import CVAnalyzer


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.max_tokens = None

    async def text_completion(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> str:
        self.calls += 1
        self.max_tokens = max_tokens
        return "The candidate is a moderate fit based on [document:0]."


class NoopEnricher:
    async def fetch(self, url: str):
        raise AssertionError(f"Unexpected URL enrichment: {url}")


class FakeJobPostingClient:
    async def fetch(self, job_posting_id: str):
        return SimpleNamespace(
            id=job_posting_id,
            title="Backend Engineer",
            description="Build reliable Python APIs.",
        )


def settings(**overrides):
    values = {
        "cv_max_files": 3,
        "cv_max_file_size_bytes": 1_000_000,
        "cv_max_total_size_bytes": 2_000_000,
        "cv_max_pages": 10,
        "cv_max_extracted_chars": 10_000,
        "cv_llm_evidence_chars": 5_000,
        "cv_response_evidence_chars": 5_000,
        "pdf_extraction_workers": 2,
        "pdf_process_budget": 2,
        "extraction_concurrency": 2,
        "docx_max_entries": 1_000,
        "docx_max_uncompressed_bytes": 50_000_000,
        "docx_max_compression_ratio": 100,
        "scrape_max_links": 10,
        "scrape_concurrency": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def analyzer(**overrides):
    return CVAnalyzer(
        settings(**overrides),
        FakeLLM(),
        NoopEnricher(),
        NoopEnricher(),
        FakeJobPostingClient(),
    )


def test_source_budget_preserves_all_metadata_and_shares_excerpt_space():
    sources = [
        {
            "id": f"web:{index}",
            "url": f"https://example.com/{index}",
            "type": "website",
            "title": f"Source {index}",
            "excerpt": "x" * 10_000,
        }
        for index in range(5)
    ]
    budgeted, truncated = CVAnalyzer._budget_sources(sources, 5_000)
    assert [source["id"] for source in budgeted] == [source["id"] for source in sources]
    assert all(source["excerpt"] for source in budgeted)
    assert (
        max(len(source["excerpt"]) for source in budgeted)
        - min(len(source["excerpt"]) for source in budgeted)
        <= 1
    )
    assert len(json.dumps(budgeted, ensure_ascii=False)) <= 5_000
    assert truncated


def test_document_budget_uses_upload_identity_for_duplicate_names():
    service = analyzer(cv_max_extracted_chars=180)
    text, included_ids, truncated = service._build_cv(
        [(0, "same.pdf", "A" * 100), (1, "same.pdf", "B" * 100)]
    )
    assert included_ids == {0, 1}
    assert "DOCUMENT 0" in text and "DOCUMENT 1" in text
    assert "A" in text and "B" in text
    assert truncated


def test_summary_is_display_ready_without_internal_references():
    analysis = (
        "Stefanus is a moderate fit [document:0].\n"
        "Backend experience should be verified [job_description], while the portfolio "
        "is frontend-heavy [github:StefanusTitan/lifetime-art]."
    )

    summary = CVAnalyzer._prepare_summary(analysis)

    assert summary == (
        "Stefanus is a moderate fit.\n"
        "Backend experience should be verified, while "
        "the portfolio is frontend-heavy."
    )


def test_summary_normalizes_model_markdown_to_simple_html():
    analysis = """**Status Kesesuaian:** Sedang

**Kekuatan Utama Kandidat:**
* **Frontend kuat:** React dan TypeScript
* Pengalaman *Agile/Scrum*

**Rekomendasi:** Lanjutkan wawancara."""

    summary = CVAnalyzer._prepare_summary(analysis)

    assert summary == (
        "<p><b>Status Kesesuaian:</b> Sedang</p>"
        "<p><b>Kekuatan Utama Kandidat:</b></p>"
        "<ul><li><b>Frontend kuat:</b> React dan TypeScript</li>"
        "<li>Pengalaman <i>Agile/Scrum</i></li></ul>"
        "<p><b>Rekomendasi:</b> Lanjutkan wawancara.</p>"
    )
    assert "**" not in summary


def test_analysis_prompt_requires_html_and_forbids_markdown():
    prompt = CVAnalyzer._analysis_prompt()

    assert "hanya fragmen HTML" in prompt
    assert "bukan Markdown" in prompt
    assert "<p><b>Status Kesesuaian:</b>" in prompt
    assert "mudah dipahami orang nonteknis" in prompt
    assert "maksimal 120 kata" in prompt
    assert "Tepat dua poin" in prompt
    assert "Jelaskan dampak setiap pengalaman atau keahlian" in prompt
    assert "<p><b>Rekomendasi untuk HR:</b>" in prompt
    assert "<p><b>Pertanyaan Wawancara yang Disarankan:</b>" in prompt


def test_analyzer_returns_narrative_and_closes_upload():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Candidate built a production Python API")
    data = document.tobytes()
    document.close()
    upload = UploadFile(filename="candidate.pdf", file=io.BytesIO(data))
    service = analyzer()

    job_posting_id = "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff"
    result = asyncio.run(service.analyze(job_posting_id, [upload]))

    assert result.job_posting_id == job_posting_id
    assert result.job_title == "Backend Engineer"
    assert "moderate fit" in result.analysis
    assert "[document:0]" not in result.analysis
    assert result.sources[0].id == "document:0"
    assert service.llm.calls == 1
    assert service.llm.max_tokens == 500
    assert upload.file.closed
