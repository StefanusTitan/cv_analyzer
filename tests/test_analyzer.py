import asyncio
import io
import json
from types import SimpleNamespace

from fastapi import UploadFile
from pdf_oxide import Pdf

from app.core.errors import UpstreamError
from app.services import cv_analyzer as cv_analyzer_module
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
        "llm_max_output_tokens": 8_000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def analyzer(**overrides):
    return CVAnalyzer(
        settings(**overrides),
        FakeLLM(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
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


def test_analysis_sources_keep_documents_and_external_excerpts():
    sources = [
        {
            "id": "document:0",
            "type": "document",
            "excerpt": None,
        },
        {
            "id": "github:profile",
            "type": "github",
            "excerpt": json.dumps({"bio": "Engineer", "public_repos": 10}),
        },
        {
            "id": "github:thin",
            "type": "github",
            "excerpt": json.dumps({"language": "TypeScript", "updated_at": "2026"}),
        },
        {
            "id": "github:description-only",
            "type": "github",
            "excerpt": json.dumps({"description": "Production API"}),
        },
        {
            "id": "github:empty",
            "type": "github",
            "excerpt": "",
        },
        {
            "id": "github:useful",
            "type": "github",
            "excerpt": json.dumps({"readme": "Production API documentation"}),
        },
        {
            "id": "web:useful",
            "type": "website",
            "excerpt": "Public project documentation",
        },
        {
            "id": "web:empty",
            "type": "website",
            "excerpt": None,
        },
    ]

    selected = CVAnalyzer._analysis_sources(sources)

    assert [source["id"] for source in selected] == [
        "document:0",
        "github:profile",
        "github:thin",
        "github:description-only",
        "github:useful",
        "web:useful",
    ]


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
    assert "token sumber seperti [[S1]]" in prompt
    assert "token dokumen" in prompt
    assert "tidak membuktikan pengalaman kerja" in prompt
    assert "tanyakan waktu mulai yang diinginkan secara netral" in prompt
    assert "Jangan mengaitkan teknologi proyek dengan pengalaman kerja" in prompt
    assert "bukan membuktikan kemahiran atau kualitas" in prompt
    assert "Manifest hanya menunjukkan dependensi dan skrip" in prompt
    assert "satu alasan singkat [[S1]]" in prompt
    assert "Persyaratan di JOB DESCRIPTION bukan bukti pengalaman kandidat" in prompt
    assert "Hindari kata menguasai" in prompt
    assert "jangan membuat token baru" in prompt
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


def test_bilingual_summary_resolves_trusted_source_tokens_without_rejecting_gaps():
    sources = [
        {
            "id": "document:0",
            "url": "document://0/candidate-resume.pdf",
            "type": "document",
            "title": "candidate-resume.pdf",
        },
        {
            "id": "github:candidate/project",
            "url": "https://github.com/candidate/project",
            "type": "github",
            "title": "candidate/project",
        },
        {
            "id": "github:candidate/other",
            "url": "https://github.com/candidate/other",
            "type": "github",
            "title": "candidate/other",
        },
    ]
    raw = json.dumps(
        {
            "id": (
                "<ul><li>Pengalaman produksi [[S1]].</li>"
                "<li>Proyek publik [[S2]][[S3]] [[S99]] https://example.com/invented.</li></ul>"
            ),
            "en": (
                "<ul><li>Production experience [[S1]].</li>"
                "<li>Public project [[S2]][[S3]] [[S99]] https://example.com/invented.</li></ul>"
            ),
        }
    )

    indonesian, english = CVAnalyzer._prepare_bilingual_summary(raw, sources)

    assert "Pengalaman produksi [Resume]." in indonesian
    assert "Production experience [Resume]." in english
    assert "https://github.com/candidate/project" in indonesian
    assert "https://github.com/candidate/project" in english
    assert "https://github.com/candidate/project https://github.com/candidate/other" in indonesian
    assert "https://github.com/candidate/project https://github.com/candidate/other" in english
    assert "S99" not in indonesian
    assert "S99" not in english
    assert "example.com" not in indonesian
    assert "example.com" not in english


def test_summary_repairs_stray_list_closing_tag_after_heading():
    summary = CVAnalyzer._prepare_summary(
        "<p><b>What to confirm:</b></ul><ul><li>Testing depth.</li></ul>"
    )

    assert summary == (
        "<p><b>What to confirm:</b></p><ul><li>Testing depth.</li></ul>"
    )


def test_document_citation_labels_number_repeated_types_only():
    sources = [
        {
            "id": "document:0",
            "url": "document://0/first-cv.pdf",
            "type": "document",
            "title": "first-cv.pdf",
        },
        {
            "id": "document:1",
            "url": "document://1/resume.pdf",
            "type": "document",
            "title": "resume.pdf",
        },
        {
            "id": "document:2",
            "url": "document://2/second-cv.pdf",
            "type": "document",
            "title": "second-cv.pdf",
        },
    ]

    assert CVAnalyzer._citation_replacements(sources) == {
        "1": "[CV 1]",
        "2": "[Resume]",
        "3": "[CV 2]",
    }


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

    assert "[[S1]] WEBSITE: https://example.com/profile" in evidence
    assert "--- WEBSITE [[S1]] ---" in evidence
    assert "SOURCE URL: https://example.com/profile" in evidence
    assert "Gunakan token sumber" in evidence
    assert "Candidate Profile" not in evidence


def test_github_readme_is_included_in_llm_evidence():
    excerpt = json.dumps(
        {
            "description": "CLI tool",
            "readme": "pip install example-cli",
            "package": {
                "name": "example-cli",
                "dependencies": ["next", "react"],
                "scripts": ["build", "test"],
            },
            "tags": ["text-classification", "language:en"],
        },
        ensure_ascii=False,
    )

    formatted = CVAnalyzer._format_excerpt(excerpt, "github")

    assert "README:" in formatted
    assert "pip install example-cli" in formatted
    assert "PACKAGE.JSON:" in formatted
    assert "declared dependencies: next, react" in formatted
    assert "Tags: text-classification, language:en" in formatted


def test_enrichment_dedupes_listed_and_linked_repos_preferring_full_data():
    class SplitGithub:
        async def fetch(self, url: str):
            if url.rstrip("/").endswith("octocat"):
                return [
                    {
                        "id": "github:octocat",
                        "url": url,
                        "type": "github",
                        "title": "Octo",
                        "excerpt": "profile",
                    },
                    {
                        "id": "github:octocat/hello",
                        "url": "https://github.com/octocat/hello",
                        "type": "github",
                        "title": "hello",
                        "excerpt": "listed",
                    },
                ]
            return [
                {
                    "id": "github:octocat/hello",
                    "url": url,
                    "type": "github",
                    "title": "hello",
                    "excerpt": "f" * 400,
                }
            ]

    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        SplitGithub(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        FakeJobPostingClient(),
    )
    warnings: list[str] = []

    sources, outcomes = asyncio.run(
        service._enrich_all(
            ["https://github.com/octocat", "https://github.com/octocat/hello"],
            warnings,
        )
    )

    hello = [source for source in sources if source["id"] == "github:octocat/hello"]
    assert len(hello) == 1
    assert len(hello[0]["excerpt"]) == 400
    assert any(source["id"] == "github:octocat" for source in sources)
    assert outcomes == {"succeeded": 2}


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
    assert "Analysis completed without supported source citations" in result.warnings
    assert service.llm.calls == 1
    assert service.llm.max_tokens == service.settings.llm_max_output_tokens
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
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
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


def test_bare_github_reference_is_discovered_and_enriched():
    class RecordingGithub:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return [
                {
                    "id": "github:octocat",
                    "url": url,
                    "type": "github",
                    "title": "Octo",
                    "excerpt": json.dumps({"bio": "Builder"}),
                }
            ]

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

    data = Pdf.from_text("GitHub: github.com/octocat").to_bytes()
    upload = UploadFile(filename="cv.pdf", file=io.BytesIO(data))
    github = RecordingGithub()
    llm = CapturingLLM()
    service = CVAnalyzer(
        settings(),
        llm,
        github,
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    assert github.urls == ["https://github.com/octocat"]
    assert "github:octocat" in {source.id for source in result.sources}
    assert any(source.type == "github" for source in result.sources)
    assert "Builder" in llm.user_prompt
    assert "[[S2]]" in llm.user_prompt


def test_bare_gitlab_reference_is_discovered_and_structurally_enriched():
    class RecordingGitlab:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return [
                {
                    "id": "gitlab:candidate/project",
                    "url": url,
                    "type": "gitlab",
                    "title": "candidate/project",
                    "excerpt": json.dumps(
                        {"description": "Public service", "languages": {"Go": 100}}
                    ),
                }
            ]

    data = Pdf.from_text("Code: gitlab.com/candidate/project").to_bytes()
    upload = UploadFile(filename="cv.pdf", file=io.BytesIO(data))
    gitlab = RecordingGitlab()
    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        NoopEnricher(),
        gitlab,
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    assert gitlab.urls == ["https://gitlab.com/candidate/project"]
    source = next(source for source in result.sources if source.type == "gitlab")
    assert source.kind == "repository"
    assert source.access_status == "public"


def test_behance_project_uses_cross_role_artifact_metadata():
    class RecordingScraper:
        async def fetch(self, url: str):
            return {
                "id": "web:behance-project",
                "url": url,
                "type": "website",
                "title": "Campaign Case Study",
                "excerpt": "Brand strategy, contribution, and campaign outcomes",
            }

    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        RecordingScraper(),
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze(
            "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff",
            [],
            links=["behance.net/gallery/123/Campaign-Case-Study"],
        )
    )

    source = next(source for source in result.sources if source.type == "behance")
    assert source.kind == "project"
    assert source.access_status == "public"


def test_bitbucket_and_hugging_face_use_structured_enrichers():
    class RecordingEnricher:
        def __init__(self, source_type: str):
            self.source_type = source_type
            self.urls = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return [
                {
                    "id": f"{self.source_type}:candidate/work",
                    "url": url,
                    "type": self.source_type,
                    "title": "candidate/work",
                    "excerpt": "Public metadata",
                }
            ]

    class RejectingScraper:
        async def fetch(self, url: str):
            raise AssertionError(f"Structured URL reached scraper: {url}")

    bitbucket = RecordingEnricher("bitbucket")
    huggingface = RecordingEnricher("huggingface")
    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        NoopEnricher(),
        NoopEnricher(),
        bitbucket,
        huggingface,
        NoopEnricher(),
        RejectingScraper(),
        FakeJobPostingClient(),
    )

    async def run():
        warnings = []
        bitbucket_result = await service._enrich(
            "https://bitbucket.org/candidate/work", warnings
        )
        huggingface_result = await service._enrich(
            "https://huggingface.co/candidate/work", warnings
        )
        return bitbucket_result, huggingface_result, warnings

    bitbucket_result, huggingface_result, warnings = asyncio.run(run())

    assert bitbucket.urls == ["https://bitbucket.org/candidate/work"]
    assert huggingface.urls == ["https://huggingface.co/candidate/work"]
    assert bitbucket_result[1] == "succeeded"
    assert huggingface_result[1] == "succeeded"
    assert warnings == []


def test_media_routing_and_telemetry_exclude_candidate_urls(monkeypatch):
    class RecordingOEmbed:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return [
                {
                    "id": "youtube:video",
                    "url": url,
                    "type": "youtube",
                    "kind": "video",
                    "access_status": "public",
                    "title": "Candidate presentation",
                    "excerpt": json.dumps({"author": "Candidate"}),
                }
            ]

    class RecordingLogger:
        def __init__(self):
            self.extra = None

        def bind(self, **extra):
            self.extra = extra
            return self

        def info(self, message: str):
            return None

    oembed = RecordingOEmbed()
    recording_logger = RecordingLogger()
    monkeypatch.setattr(cv_analyzer_module, "logger", recording_logger)
    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        oembed,
        NoopEnricher(),
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze(
            "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff",
            [],
            links=[
                "https://youtu.be/private-candidate-path",
                "https://linkedin.com/in/private-candidate-path",
            ],
        )
    )

    assert oembed.urls == ["https://youtu.be/private-candidate-path"]
    assert any(source.type == "youtube" for source in result.sources)
    telemetry = recording_logger.extra["enrichment"]
    assert telemetry == {
        "submitted_platforms": {"linkedin": 1, "youtube": 1},
        "enriched_platforms": {"youtube": 1},
        "outcomes": {"restricted": 1, "succeeded": 1},
    }
    assert "private-candidate-path" not in json.dumps(telemetry)


def test_llm_input_is_logged_before_completion(monkeypatch):
    class RecordingLogger:
        def __init__(self):
            self.records: list[dict] = []

        def bind(self, **extra):
            self.records.append({"extra": extra})
            return self

        def info(self, message: str):
            if self.records:
                self.records[-1]["message"] = message

    recording_logger = RecordingLogger()
    monkeypatch.setattr(cv_analyzer_module, "logger", recording_logger)
    data = Pdf.from_text("Candidate built a production Python API").to_bytes()
    upload = UploadFile(filename="candidate.pdf", file=io.BytesIO(data))
    service = analyzer()

    asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    llm_input = next(
        record
        for record in recording_logger.records
        if record["extra"].get("event") == "llm_input"
    )
    llm = llm_input["extra"]["llm"]
    assert llm_input["message"] == "LLM analysis input"
    assert "Kamu adalah asisten rekrutmen" in llm["system"]
    assert "JOB TITLE: Backend Engineer" in llm["user"]
    assert "Candidate built a production Python API" in llm["user"]
    assert "=== CV / RESUME ===" in llm["user"]
    assert llm["system_chars"] == len(llm["system"])
    assert llm["user_chars"] == len(llm["user"])
    assert llm["max_output"] == service.settings.llm_max_output_tokens


def test_github_mentions_without_profile_path_are_not_enriched():
    class RecordingGithub:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return []

    class RecordingScraper:
        def __init__(self):
            self.urls: list[str] = []

        async def fetch(self, url: str):
            self.urls.append(url)
            return {
                "id": "web:gist",
                "url": url,
                "type": "website",
                "title": "Gist",
                "excerpt": "Public code sample",
            }

    data = Pdf.from_text(
        "Email someone@github.com or see gist.github.com/octocat/abc"
    ).to_bytes()
    upload = UploadFile(filename="cv.pdf", file=io.BytesIO(data))
    github = RecordingGithub()
    scraper = RecordingScraper()
    service = CVAnalyzer(
        settings(),
        FakeLLM(),
        github,
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        scraper,
        FakeJobPostingClient(),
    )

    asyncio.run(service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload]))

    assert github.urls == []
    assert scraper.urls == ["https://gist.github.com/octocat/abc"]


def test_anchor_citations_are_unwrapped_to_plain_urls():
    analysis = (
        "<p><b>Alasan Kandidat Cocok:</b></p>"
        "<ul><li>Strong C2 tooling evidence "
        '<a href="https://github.com/octocat/c2-tool">https://github.com/octocat/c2-tool</a>.</li>'
        "<li>Public work under a "
        '<a href="https://github.com/octocat">different label</a>.</li></ul>'
    )

    summary = CVAnalyzer._prepare_summary(analysis)

    assert "<a" not in summary
    assert "https://github.com/octocat/c2-tool." in summary
    assert "https://github.com/octocat.</li>" in summary
    assert "different label" not in summary


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
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        NoopEnricher(),
        scraper,
        FakeJobPostingClient(),
    )

    result = asyncio.run(
        service.analyze("32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff", [upload])
    )

    assert scraper.urls == []
    assert any("website_access_restricted" in warning for warning in result.warnings)
    linkedin_source = next(
        source for source in result.sources if source.type == "linkedin"
    )
    assert linkedin_source.kind == "profile"
    assert linkedin_source.access_status == "restricted"
    assert service.llm.calls == 1
