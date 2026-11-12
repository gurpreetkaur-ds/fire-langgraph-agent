"""Upload validation and text extraction for PDF, DOCX, TXT, Markdown and CSV.

Validation never trusts the client: the extension must be allowed AND the bytes must look like that type
(PDF magic, DOCX = a sane ZIP containing word/document.xml, text types = decodable text without NUL bytes).
Extraction returns `Section`s that keep provenance (page number for PDFs, heading for Markdown/DOCX, row range
for CSV) so chunks and citations can point back to where the text came from.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..settings import ALLOWED_TYPES

MAX_DOCX_UNCOMPRESSED = 120 * 1024 * 1024  # zip-bomb guard
CSV_ROWS_PER_SECTION = 25


class UploadRejected(ValueError):
    """The file is not acceptable; the message is safe to show to the user."""


class ExtractionError(ValueError):
    """The file was accepted but no usable text could be extracted; message is user-safe."""


@dataclass
class Section:
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None


@dataclass
class ExtractedDocument:
    sections: List[Section]
    page_count: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


# ---- validation ------------------------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """Display name only (files are stored under a generated id, never this name)."""
    name = Path((name or "").replace("\\", "/")).name
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*').strip(" .")
    return (name or "document")[:200]


def detect_type(filename: str, data: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    ftype = ALLOWED_TYPES.get(ext)
    if ftype is None:
        allowed = ", ".join(sorted({e.lstrip(".").upper() for e in ALLOWED_TYPES}))
        raise UploadRejected(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")
    if not data:
        raise UploadRejected("The file is empty.")
    if ftype == "pdf":
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise UploadRejected("This file does not look like a PDF.")
    elif ftype == "docx":
        _check_docx_zip(data)
    else:
        decode_text(data)  # raises UploadRejected for binary content
    return ftype


def _check_docx_zip(data: bytes) -> None:
    if not data.startswith(b"PK\x03\x04"):
        raise UploadRejected("This file does not look like a DOCX document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "word/document.xml" not in names:
                raise UploadRejected("This ZIP file is not a Word (.docx) document.")
            if sum(i.file_size for i in zf.infolist()) > MAX_DOCX_UNCOMPRESSED:
                raise UploadRejected("This DOCX expands to an unreasonably large size and was rejected.")
    except zipfile.BadZipFile as exc:
        raise UploadRejected("This DOCX file is corrupted.") from exc


def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise UploadRejected("Could not decode this UTF-16 text file.") from exc
    if b"\x00" in data[:65536]:
        raise UploadRejected("This file contains binary data, not text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


# ---- cleaning --------------------------------------------------------------------------------------------

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across PDF line breaks
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NL.sub("\n\n", text).strip()


# ---- extraction ------------------------------------------------------------------------------------------

def extract(path: Path, ftype: str, *, max_pdf_pages: int = 600) -> ExtractedDocument:
    data = path.read_bytes()
    if ftype == "pdf":
        doc = _extract_pdf(data, max_pdf_pages)
    elif ftype == "docx":
        doc = _extract_docx(data)
    elif ftype == "md":
        doc = _extract_markdown(decode_text(data))
    elif ftype == "csv":
        doc = _extract_csv(decode_text(data))
    else:
        doc = ExtractedDocument(sections=[Section(clean_text(decode_text(data)))])
    doc.sections = [s for s in doc.sections if s.text.strip()]
    if not doc.sections:
        hint = " It may be a scanned PDF (images only); OCR is not supported." if ftype == "pdf" else ""
        raise ExtractionError("No readable text was found in this file." + hint)
    return doc


def _extract_pdf(data: bytes, max_pages: int) -> ExtractedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ExtractionError("This PDF is password-protected.")
            except ExtractionError:
                raise
            except Exception as exc:
                raise ExtractionError("This PDF is password-protected.") from exc
        n = len(reader.pages)
    except PdfReadError as exc:
        raise ExtractionError("This PDF could not be read (it may be corrupted).") from exc
    if n > max_pages:
        raise ExtractionError(f"This PDF has {n} pages; the limit is {max_pages}.")
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # one bad page should not sink the document
            text = ""
        sections.append(Section(clean_text(text), page=i))
    meta = {}
    info = reader.metadata or {}
    for key, name in (("/Title", "title"), ("/Author", "author"), ("/Subject", "subject")):
        val = info.get(key)
        if val and str(val).strip():
            meta[name] = str(val).strip()[:300]
    return ExtractedDocument(sections=sections, page_count=n, meta=meta)


def _extract_docx(data: bytes) -> ExtractedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError("This DOCX file could not be read.") from exc

    sections: List[Section] = []
    heading: Optional[str] = None
    buf: List[str] = []

    def flush():
        if buf:
            sections.append(Section(clean_text("\n".join(buf)), heading=heading))
            buf.clear()

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            style = (block.style.name or "").lower() if block.style is not None else ""
            if text and (style.startswith("heading") or style == "title"):
                flush()
                heading = text[:300]
                buf.append(text)
            elif text:
                buf.append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                # merged cells repeat their text; drop consecutive duplicates
                dedup = [c for j, c in enumerate(cells) if c and (j == 0 or c != cells[j - 1])]
                if dedup:
                    buf.append(" | ".join(dedup))
    flush()
    props = document.core_properties
    meta = {k: str(v).strip()[:300] for k, v in (("title", props.title), ("author", props.author), ("subject", props.subject)) if v and str(v).strip()}
    return ExtractedDocument(sections=sections, meta=meta)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _extract_markdown(text: str) -> ExtractedDocument:
    sections: List[Section] = []
    heading: Optional[str] = None
    buf: List[str] = []
    in_fence = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            if buf:
                sections.append(Section(clean_text("\n".join(buf)), heading=heading))
                buf = []
            heading = m.group(2).strip()[:300]
        buf.append(line)
    if buf:
        sections.append(Section(clean_text("\n".join(buf)), heading=heading))
    title = next((s.heading for s in sections if s.heading), None)
    return ExtractedDocument(sections=sections, meta={"title": title} if title else {})


def _looks_like_header(row: List[str]) -> bool:
    cells = [c.strip() for c in row]
    if not cells or any(not c for c in cells) or len(set(cells)) != len(cells):
        return False
    return not any(re.fullmatch(r"[-+]?[\d.,%$€£ ]+", c) for c in cells)


def _extract_csv(text: str) -> ExtractedDocument:
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return ExtractedDocument(sections=[])
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True
    if not has_header and _looks_like_header(rows[0]):
        has_header = True  # the sniffer's type check is fooled by mixed columns (e.g. "custom" among prices)
    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])] if has_header else None
    body = rows[1:] if has_header else rows
    if header is None:
        width = max(len(r) for r in body)
        header = [f"column_{i + 1}" for i in range(width)]
    sections = []
    for start in range(0, len(body), CSV_ROWS_PER_SECTION):
        group = body[start:start + CSV_ROWS_PER_SECTION]
        lines = []
        for offset, row in enumerate(group):
            pairs = [f"{header[i] if i < len(header) else f'column_{i + 1}'}: {v.strip()}" for i, v in enumerate(row) if v.strip()]
            lines.append(f"Row {start + offset + 1}: " + "; ".join(pairs))
        first, last = start + 1, start + len(group)
        sections.append(Section(clean_text("\n".join(lines)), heading=f"Rows {first}-{last}"))
    return ExtractedDocument(sections=sections, meta={"columns": header[:100], "row_count": len(body)})
