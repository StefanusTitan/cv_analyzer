import asyncio
import io
import json
from types import SimpleNamespace

from fastapi import UploadFile
from pdf_oxide import Pdf

from app.core.errors import UpstreamError
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
        return json.dumps(
            {
                "id": "Kandidat cukup sesuai berdasarkan [document:0].",
                "en": "The candidate is a moderate fit based on [document:0].",
            },
            ensure_ascii=False,
        )


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
        "pdf_extraction_workers": 2,
        "pdf_process_budget": 2,
        "extraction_concurrency": 2,
        "docx_max_entries": 1_000,
        "docx_max_uncompressed_bytes": 50_000_000,
        "docx_max_compression_ratio": 100,
        "scrape_max_links": 10,
        "scrape_concurrency": 2,
        "enrichment_budget_seconds": 8.0,
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

    assert "SATU objek JSON" in prompt
    assert "tanpa Markdown" in prompt
    assert "<p><b>Status Kesesuaian:</b>" in prompt
    assert "Kualifikasi: Berlebih / Kurang / Sesuai" in prompt
    assert "Heading en: Fit," in prompt
    assert "Qualification: Overqualified / Underqualified / Just right." in prompt
    assert "akumulasi tahun pengalaman kerja" in prompt
    assert "jika disebutkan" in prompt
    assert "jangan mengarang ambang tahun" in prompt
    assert "Maksimal 300 kata per bahasa" in prompt
    assert "Dua sampai tiga poin" in prompt
    assert "dampak pengalaman terhadap pekerjaan" in prompt
    assert "<p><b>Rekomendasi untuk HR:</b>" in prompt
    assert "Pertanyaan Wawancara yang Disarankan" not in prompt
    assert "SOURCE URL" in prompt
    assert "mengarang URL" in prompt
    assert "[GitHub]" in prompt
    assert "sebagai pengganti URL" in prompt
    assert "data tidak tepercaya" in prompt


def test_bilingual_summary_accepts_fenced_json_and_rejects_missing_language():
    prepared = CVAnalyzer._prepare_bilingual_summary(
        """```json
{"id": "Kandidat cukup sesuai [document:0].", "en": "Moderate fit [document:0]."}
```"""
    )
    assert prepared == ("Kandidat cukup sesuai.", "Moderate fit.")

    try:
        CVAnalyzer._prepare_bilingual_summary('{"id": "Hanya Indonesia."}')
    except UpstreamError as exc:
        assert exc.code == "llm_invalid_response"
    else:
        raise AssertionError("expected llm_invalid_response")


def test_external_evidence_input_identifies_sources_by_url():
    evidence = CVAnalyzer._format_llm_input(
        "job-id",
        "Backend Engineer",
        "Build reliable Python APIs.",
        "Candidate built APIs.",
        [
            {
                "id": "web:1",
                "url": "https://example.com/profile",
                "type": "website",
                "title": "Candidate Profile",
                "excerpt": "Built production APIs.",
            }
        ],
    )

    assert "SOURCE URL: https://example.com/profile" in evidence
    assert "bukan judul, nama sumber, atau label" in evidence
    assert "Candidate Profile" not in evidence


def test_analyzer_returns_narrative_and_closes_upload():
    data = Pdf.from_text("Candidate built a production Python API").to_bytes()
    upload = UploadFile(filename="candidate.pdf", file=io.BytesIO(data))
    service = analyzer()

    job_posting_id = "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff"
    result = asyncio.run(service.analyze(job_posting_id, [upload]))

    assert result.job_posting_id == job_posting_id
    assert result.job_title == "Backend Engineer"
    assert "cukup sesuai" in result.analysis
    assert "moderate fit" in result.analysis_en
    assert "[document:0]" not in result.analysis
    assert "[document:0]" not in result.analysis_en
    assert result.sources[0].id == "document:0"
    assert service.llm.calls == 1
    assert service.llm.max_tokens == 2500
    assert upload.file.closed


def test_enrichment_budget_keeps_finished_sources_and_continues():
    class SlowScraper:
        def __init__(self):
            self.started = 0

        async def fetch(self, url: str):
            self.started += 1
            if "fast" in url:
                return {
                    "id": "web:fast",
                    "url": url,
                    "type": "website",
                    "title": "Fast",
                    "excerpt": "ok",
                }
            await asyncio.sleep(2)
            return {
                "id": "web:slow",
                "url": url,
                "type": "website",
                "title": "Slow",
                "excerpt": "late",
            }

    class CapturingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.user_prompt = ""

        async def text_completion(
            self, system: str, user: str, *, max_tokens: int | None = None
        ) -> str:
            self.user_prompt = user
            return await super().text_completion(
                system, user, max_tokens=max_tokens
            )

    data = Pdf.from_text(
        "See https://fast.example.test/profile and https://slow.example.test/blog"
    ).to_bytes()
    upload = UploadFile(filename="links.pdf", file=io.BytesIO(data))

    scraper = SlowScraper()
    llm = CapturingLLM()
    service = CVAnalyzer(
        settings(enrichment_budget_seconds=0.3, scrape_max_links=10),
        llm,
        NoopEnricher(),
        scraper,
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    assert service.llm.calls == 1
    assert any("time budget" in warning for warning in result.warnings)
    assert any("enrichment_budget_exceeded" in warning for warning in result.warnings)
    assert "web:fast" in {source.id for source in result.sources}
    assert "web:slow" not in {source.id for source in result.sources}
    assert "Fast" in llm.user_prompt or "fast.example.test" in llm.user_prompt


def test_linkedin_urls_are_not_sent_to_scraper():
    class RecordingScraper:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            raise AssertionError("LinkedIn should be skipped before scrape")

    data = Pdf.from_text(
        "Profile https://www.linkedin.com/in/candidate"
    ).to_bytes()
    upload = UploadFile(filename="li.pdf", file=io.BytesIO(data))
    scraper = RecordingScraper()
    service = CVAnalyzer(
        settings(scrape_max_links=5),
        FakeLLM(),
        NoopEnricher(),
        scraper,
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    assert scraper.urls == []
    assert any("website_access_restricted" in warning for warning in result.warnings)
    assert service.llm.calls == 1
