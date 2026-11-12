"""Retrieval post-processing: hybrid reranking, near-duplicate filtering, context building and citation validation.

All deterministic (no LLM), so it adds milliseconds, not seconds, and is fully unit-testable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .store import Hit

_WORD = re.compile(r"[a-z0-9][a-z0-9_\-']*")
STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in into is it its me my no not "
    "of on or our shall should so than that the their them then there these they this to was we were what when where "
    "which who whom why will with would you your about tell explain give list according document documents file says "
    "say please".split()
)
VECTOR_WEIGHT, LEXICAL_WEIGHT = 0.8, 0.2
DUPLICATE_JACCARD = 0.8


def terms(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS]


def _shingles(text: str, n: int = 5) -> set:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class Ranked:
    hit: Hit
    vector_score: float
    lexical_score: float
    score: float


def rerank(query: str, hits: Sequence[Hit], *, top_k: int, threshold: float,
           relative_cutoff: float = 0.0) -> Tuple[List[Ranked], Dict[str, int]]:
    """Hybrid rerank (cosine + IDF-weighted term overlap), drop weak and near-duplicate chunks.

    - absolute threshold: cosine similarity below `threshold` is never used
    - relative cutoff: chunks scoring below `relative_cutoff` x the best chunk's similarity are dropped, so a clearly
      best passage is not diluted by a tail of loosely related ones (0 disables; the best chunk is always kept)
    Returns the kept chunks (best first) and counts of what was dropped, for observability.
    """
    stats = {"candidates": len(hits), "below_threshold": 0, "below_relative": 0, "duplicates": 0}
    above = [h for h in hits if h.score >= threshold]
    stats["below_threshold"] = len(hits) - len(above)
    if not above:
        return [], stats
    if relative_cutoff > 0:
        best = max(h.score for h in above)
        strong = [h for h in above if h.score >= best * relative_cutoff]
        stats["below_relative"] = len(above) - len(strong)
        above = strong

    q_terms = set(terms(query))
    doc_terms = [set(terms(h.text)) for h in above]
    df = Counter(t for ts in doc_terms for t in ts if t in q_terms)
    n = len(above)
    idf = {t: math.log(1 + (n + 1) / (df.get(t, 0) + 0.5)) for t in q_terms}
    denom = sum(idf.values()) or 1.0

    ranked = []
    for h, ts in zip(above, doc_terms):
        lex = sum(idf[t] for t in q_terms if t in ts) / denom if q_terms else 0.0
        ranked.append(Ranked(h, h.score, lex, VECTOR_WEIGHT * h.score + LEXICAL_WEIGHT * lex))
    ranked.sort(key=lambda r: r.score, reverse=True)

    kept: List[Ranked] = []
    kept_shingles: List[set] = []
    for r in ranked:
        sh = _shingles(r.hit.text)
        if any(jaccard(sh, other) >= DUPLICATE_JACCARD for other in kept_shingles):
            stats["duplicates"] += 1
            continue
        kept.append(r)
        kept_shingles.append(sh)
        if len(kept) >= top_k:
            break
    return kept, stats


def location(hit_or_source) -> str:
    page = getattr(hit_or_source, "page", None) if not isinstance(hit_or_source, dict) else hit_or_source.get("page")
    section = getattr(hit_or_source, "section", None) if not isinstance(hit_or_source, dict) else hit_or_source.get("section")
    parts = []
    if page:
        parts.append(f"page {page}")
    if section:
        parts.append(section)
    return ", ".join(parts)


def build_context(ranked: Sequence[Ranked], char_budget: int) -> Tuple[List[dict], str]:
    """Label kept chunks S1..Sn and render the context block, staying within `char_budget` characters."""
    sources: List[dict] = []
    blocks: List[str] = []
    used = 0
    for r in ranked:
        h = r.hit
        label = f"S{len(sources) + 1}"
        loc = location(h)
        header = f"[{label}] {h.filename}" + (f" ({loc})" if loc else "")
        room = char_budget - used - len(header) - 2
        if room < 200:
            break
        body = h.text if len(h.text) <= room else h.text[:room].rsplit(" ", 1)[0] + " …"
        blocks.append(f"{header}\n{body}")
        used += len(header) + len(body) + 2
        sources.append({
            "id": label, "chunk_id": h.chunk_id, "document_id": h.document_id, "filename": h.filename,
            "file_type": h.file_type, "page": h.page, "section": h.section, "chunk_index": h.chunk_index,
            "score": round(r.score, 4), "similarity": round(r.vector_score, 4), "text": body,
        })
    return sources, "\n\n".join(blocks)


NOT_FOUND_MARKERS = ("not in your documents", "no information about", "does not contain details", "do not contain details", "don't contain", "do not contain", "doesn't contain", "does not contain",
                     "couldn't find", "could not find", "no information", "not mentioned", "not covered")


def says_not_found(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in NOT_FOUND_MARKERS)


_CITE_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]")
_CITE_NUM = re.compile(r"\d+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def validate_citations(answer: str, sources: Sequence[dict]) -> dict:
    """Check every [S#] marker against the sources actually provided.

    - Markers that point to a source that was not retrieved are removed from the answer (never shown).
    - Each valid marker yields a {claim, source_id} citation, the claim being its sentence without markers.
    Returns the cleaned answer, citations, used/unused source ids and the ids of removed markers.
    """
    valid_ids = {s["id"] for s in sources}
    invalid: List[str] = []

    def fix_group(m: re.Match) -> str:
        ids = [f"S{n}" for n in _CITE_NUM.findall(m.group(0))]
        good = list(dict.fromkeys(i for i in ids if i in valid_ids))
        invalid.extend(i for i in ids if i not in valid_ids)
        return "".join(f"[{i}]" for i in good)

    cleaned = _CITE_GROUP.sub(fix_group, answer)
    # A source cannot support a claim that information is absent: drop citations from such sentences.
    parts = re.split(r"((?<=[.!?])\s+|\n+)", cleaned)
    absence: List[str] = []
    for i, part in enumerate(parts):
        if re.search(r"\[S\d+\]", part) and says_not_found(part):
            absence += re.findall(r"\[(S\d+)\]", part)
            parts[i] = re.sub(r"\s*\[S\d+\]", "", part)
    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)

    citations: List[dict] = []
    used: List[str] = []
    previous = ""  # last text segment: a marker on a line of its own (e.g. after a table) cites it
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        ids = re.findall(r"\[(S\d+)\]", sentence)
        claim = re.sub(r"\s*\[S\d+\]", "", sentence).strip(" -*#>|\t")
        if not ids:
            if claim:
                previous = claim
            continue
        if not claim:
            claim = previous
        else:
            previous = claim
        for sid in dict.fromkeys(ids):
            if claim:
                citations.append({"claim": claim[:500], "source_id": sid})
            if sid not in used:
                used.append(sid)
    return {
        "answer": cleaned,
        "citations": citations,
        "used_sources": used,
        "unused_sources": [s["id"] for s in sources if s["id"] not in used],
        "removed_markers": invalid,
        "absence_citations_removed": absence,
    }


def preview(text: str, n: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def pick_sources_for_ui(sources: Sequence[dict], used: Optional[Sequence[str]] = None) -> List[dict]:
    used_set = set(used or [])
    return [{**{k: v for k, v in s.items() if k != "text"}, "preview": preview(s["text"]), "cited": s["id"] in used_set} for s in sources]
