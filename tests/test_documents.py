import io
import tempfile
import zipfile
from pathlib import Path

import pymupdf
import pytest

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
    document = pymupdf.open()
    for index in range(5):
        page = document.new_page()
        # Multi-line insert so enough text survives PDF layout/extraction.
        page.insert_text((72, 72), f"PAGE-{index}-\n" + ("x" * 80 + "\n") * 8)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        path = Path(handle.name)
        handle.write(document.tobytes())
    document.close()
    try:
        text = _extract_pdf_sync(str(path), workers=2, max_pages=100, max_chars=200)
        assert "PAGE-0" in text
        assert "PAGE-4" not in text
        assert text.count("[page ") < 5
    finally:
        path.unlink(missing_ok=True)
