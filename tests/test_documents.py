import io
import tempfile
import zipfile
from pathlib import Path

import pytest
from pdf_oxide import DocumentBuilder

from app.core.errors import UploadError
from app.services.document_extractor import _extract_pdf_sync, detect_kind


def test_detect_pdf():
    assert detect_kind("cv.pdf", b"%PDF-1.7") == "pdf"


def test_rejects_unsupported_format():
    with pytest.raises(UploadError) as error:
        detect_kind("cv.txt", b"hello")
    assert error.value.status_code == 415


def test_detect_docx():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("[Content_Types].xml", "xml")
        archive.writestr("word/document.xml", "xml")
    assert detect_kind("cv.docx", stream.getvalue()) == "docx"


def test_pdf_extraction_stops_once_char_budget_is_met():
    builder = DocumentBuilder()
    for index in range(5):
        page = builder.a4_page()
        # Multi-line insert so enough text survives PDF layout/extraction.
        page.text(f"PAGE-{index}-")
        page.paragraph(("x" * 80 + " ") * 8)
        page.done()
    data = builder.build()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        path = Path(handle.name)
        handle.write(data)
    try:
        text = _extract_pdf_sync(str(path), max_pages=100, max_chars=200)
        assert "PAGE-0" in text
        assert "PAGE-4" not in text
        assert text.count("[page ") < 5
    finally:
        path.unlink(missing_ok=True)
