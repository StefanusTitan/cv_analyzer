import asyncio
import io
import zipfile
from pathlib import Path

import pymupdf
from office_oxide import Document, OfficeOxideError

from app.core.errors import UploadError


def _pdf_page_count(path: str) -> int:
    try:
        with pymupdf.open(path) as document:
            return len(document)
    except Exception as exc:
        raise UploadError(
            "The PDF is corrupt, encrypted, or unreadable", 422, "invalid_document"
        ) from exc


def _pdf_page(page):
    text = page.get_text("text", sort=True)
    links = []
    for link in page.get_links():
        uri = link.get("uri")
        if uri and uri not in links:
            links.append(uri)
    if links:
        text += "\n" + "\n".join(f"[link] {uri}" for uri in links)
    return text


def _extract_pdf_sync(path: str, workers: int, max_pages: int) -> str:
    count = _pdf_page_count(path)
    if count > max_pages:
        raise UploadError(
            f"PDF exceeds maximum page count ({max_pages})", 422, "page_limit_exceeded"
        )
    method = "mp" if count > 1 and workers > 1 else "single"
    try:
        pages = pymupdf.apply_pages(
            path,
            _pdf_page,
            method=method,
            concurrency=min(workers, count),
        )
    except Exception as exc:
        raise UploadError(
            "The PDF could not be extracted", 422, "extraction_failed"
        ) from exc
    return "\n\n".join(
        f"[page {index + 1}]\n{page}" for index, page in enumerate(pages)
    )


def _extract_docx_sync(path: str) -> str:
    try:
        with Document.open(path) as document:
            return document.plain_text()
    except (OfficeOxideError, OSError, ValueError) as exc:
        raise UploadError(
            "The DOCX could not be extracted", 422, "extraction_failed"
        ) from exc


async def extract(path: Path, kind: str, workers: int, max_pages: int) -> str:
    if kind == "pdf":
        return await asyncio.to_thread(_extract_pdf_sync, str(path), workers, max_pages)
    return await asyncio.to_thread(_extract_docx_sync, str(path))


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
    raise UploadError("Only PDF and DOCX files are accepted", 415, "unsupported_format")
