import asyncio
from pathlib import Path

from pdf_oxide import PdfDocument

from app.services.document_extractor import extract

DOCS = Path(__file__).parent / "test_docs"


def test_real_pdf_fixtures_extract_text_and_links():
    pdfs = sorted(DOCS.glob("*.pdf"))
    assert len(pdfs) == 2
    for path in pdfs:
        with PdfDocument(str(path)) as document:
            assert document.page_count() >= 1
        text = asyncio.run(extract(path, "pdf", workers=2, max_pages=100))
        assert text.strip()
        assert "[page 1]" in text


def test_real_resume_extracts_embedded_urls():
    path = DOCS / "Stefanus Titan Elianto - Resume rev. 2.3.pdf"
    text = asyncio.run(extract(path, "pdf", workers=2, max_pages=100))
    assert "https://github.com/StefanusTitan" in text
    assert "https://lifetimeart-stefanus.vercel.app/" in text
