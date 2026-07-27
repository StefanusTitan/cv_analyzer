import io
import zipfile

import pytest

from app.core.errors import UploadError
from app.services.document_extractor import detect_kind


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
