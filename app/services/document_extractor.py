import asyncio
import io
import zipfile
from pathlib import Path

from office_oxide import Document, OfficeOxideError
from pdf_oxide import PdfDocument

from app.core.errors import UploadError


def _pdf_page_count(path: str) -> int:
    try:
        with PdfDocument(path) as document:
            return document.page_count()
    except Exception as exc:
        raise UploadError(
            "The PDF is corrupt, encrypted, or unreadable", 422, "invalid_document"
        ) from exc


def _pdf_page(document: PdfDocument, index: int) -> str:
    text = document.extract_text(index)
    links = []
    for annotation in document.get_annotations(index):
        if annotation.get("subtype") != "Link":
            continue
        uri = annotation.get("action_uri")
        if uri and uri not in links:
            links.append(uri)
    if links:
        text += "\n" + "\n".join(f"[link] {uri}" for uri in links)
    return text


def _extract_pdf_sync(
    path: str,
    max_pages: int,
    max_chars: int | None = None,
) -> str:
    count = _pdf_page_count(path)
    if count > max_pages:
        raise UploadError(
            f"PDF exceeds maximum page count ({max_pages})", 422, "page_limit_exceeded"
        )
    # Prefer sequential extraction when a char budget is set so we can stop once
    # enough text is collected instead of paying for every remaining page.
    use_early_stop = max_chars is not None and max_chars > 0
    try:
        with PdfDocument(path) as document:
            parts: list[str] = []
            used = 0
            limit = min(document.page_count(), max_pages)
            for index in range(limit):
                chunk = f"[page {index + 1}]\n{_pdf_page(document, index)}"
                separator = 2 if parts else 0
                parts.append(chunk)
                used += len(chunk) + separator
                if use_early_stop and used >= max_chars:
                    break
            return "\n\n".join(parts)
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError(
            "The PDF could not be extracted", 422, "extraction_failed"
        ) from exc


def _extract_office_sync(path: str) -> str:
    try:
        with Document.open(path) as document:
            return document.plain_text()
    except (OfficeOxideError, OSError, ValueError) as exc:
        raise UploadError(
            "The document could not be extracted", 422, "extraction_failed"
        ) from exc


async def extract(
    path: Path,
    kind: str,
    max_pages: int,
    max_chars: int | None = None,
) -> str:
    if kind == "pdf":
        return await asyncio.to_thread(
            _extract_pdf_sync, str(path), max_pages, max_chars
        )
    return await asyncio.to_thread(_extract_office_sync, str(path))


def _validate_docx_archive(
    data: bytes, max_entries: int, max_uncompressed: int, max_ratio: int
) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                return False
            total = 0
            for entry in entries:
                if (
                    entry.file_size > max_uncompressed
                    or entry.compress_size == 0
                    and entry.file_size > 0
                ):
                    return False
                if (
                    entry.compress_size
                    and entry.file_size / entry.compress_size > max_ratio
                ):
                    return False
                total += entry.file_size
                if total > max_uncompressed:
                    return False
            names = set(archive.namelist())
            return "[Content_Types].xml" in names and "word/document.xml" in names
    except (zipfile.BadZipFile, OSError):
        return False


def _is_ole2(data: bytes) -> bool:
    return len(data) >= 8 and data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def detect_kind(
    name: str,
    data: bytes,
    *,
    docx_max_entries: int = 1_000,
    docx_max_uncompressed_bytes: int = 50_000_000,
    docx_max_compression_ratio: int = 100,
) -> str:
    extension = Path(name or "").suffix.lower()
    if extension == ".pdf":
        if data.startswith(b"%PDF-"):
            return "pdf"
        raise UploadError("The uploaded PDF is invalid", 422, "invalid_document")
    if extension == ".doc":
        if _is_ole2(data):
            return "doc"
        raise UploadError(
            "The uploaded DOC is invalid", 422, "invalid_document"
        )
    if extension == ".docx":
        if data[:2] == b"PK" and _validate_docx_archive(
            data,
            docx_max_entries,
            docx_max_uncompressed_bytes,
            docx_max_compression_ratio,
        ):
            return "docx"
        raise UploadError(
            "The uploaded DOCX is invalid or unsafe", 422, "invalid_document"
        )
    raise UploadError("Only PDF, DOC, and DOCX files are accepted", 415, "unsupported_format")
