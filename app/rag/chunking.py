"""Split extracted sections into overlapping chunks that keep their provenance.

- Chunks never cross a section boundary, so each one has exactly one page (PDF) / heading (MD, DOCX) / row range (CSV).
- Splits prefer paragraph, then line, then sentence, then word boundaries; a hard cut is the last resort.
- Overlap (in characters) is taken from the tail of the previous chunk and aligned to a word boundary.
- char_start/char_end are offsets into the document's full text (sections joined by a blank line).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .extract import Section

SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
MIN_CHUNK_CHARS = 20
SECTION_JOINER = "\n\n"


@dataclass
class ChunkData:
    index: int
    text: str
    page: Optional[int]
    section: Optional[str]
    char_start: int
    char_end: int


def _split_spans(text: str, start: int, end: int, size: int, seps: Sequence[str]) -> List[Tuple[int, int]]:
    """Break text[start:end] into spans no longer than `size`, cutting at the coarsest separator possible."""
    if end - start <= size:
        return [(start, end)]
    for i, sep in enumerate(seps):
        cuts = []
        pos = text.find(sep, start, end)
        while pos != -1:
            cuts.append(pos + len(sep))  # separator stays with the preceding piece
            pos = text.find(sep, pos + len(sep), end)
        cuts = [c for c in cuts if start < c < end]
        if not cuts:
            continue
        spans, prev = [], start
        for c in cuts + [end]:
            if c > prev:
                spans.extend(_split_spans(text, prev, c, size, seps[i + 1:]))
            prev = c
        return spans
    return [(s, min(s + size, end)) for s in range(start, end, size)]  # no separator at all: hard cut


def _overlap_start(text: str, chunk_start: int, chunk_end: int, overlap: int) -> int:
    if overlap <= 0:
        return chunk_end
    start = max(chunk_start, chunk_end - overlap)
    if start > chunk_start:
        space = text.find(" ", start, chunk_end)
        if space != -1:
            start = space + 1
    return start


def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    """Return (start, end) spans over `text`."""
    if not text.strip():
        return []
    pieces = _split_spans(text, 0, len(text), size, SEPARATORS)
    chunks: List[Tuple[int, int]] = []
    cur_start, cur_end = pieces[0]
    for p_start, p_end in pieces[1:]:
        if p_end - cur_start <= size:
            cur_end = p_end
            continue
        chunks.append((cur_start, cur_end))
        new_start = _overlap_start(text, cur_start, cur_end, overlap)
        new_start = max(new_start, p_end - size)  # keep the next chunk within size even with overlap
        cur_start, cur_end = min(new_start, p_start), p_end
    chunks.append((cur_start, cur_end))
    return chunks


def chunk_sections(sections: Sequence[Section], size: int, overlap: int) -> List[ChunkData]:
    out: List[ChunkData] = []
    offset = 0
    for sec in sections:
        for s, e in chunk_text(sec.text, size, overlap):
            raw = sec.text[s:e]
            lead = len(raw) - len(raw.lstrip())
            body = raw.strip()
            if len(body) < MIN_CHUNK_CHARS and out:
                continue  # drop slivers (a dangling word after a split); keep one if it is all there is
            out.append(ChunkData(
                index=len(out), text=body, page=sec.page, section=sec.heading,
                char_start=offset + s + lead, char_end=offset + s + lead + len(body),
            ))
        offset += len(sec.text) + len(SECTION_JOINER)
    return out
