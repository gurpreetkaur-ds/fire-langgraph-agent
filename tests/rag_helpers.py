"""Builders for real test documents (no binary fixtures checked in) and fake LLMs for the routed graph."""
from __future__ import annotations

import io
from typing import Callable, List

from langchain_core.messages import AIMessage


def make_pdf(pages: List[str]) -> bytes:
    """A valid multi-page PDF with real text objects (Helvetica), readable by pypdf."""
    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = len(objects) + 1 + 2 * len(pages)  # reserved after page/content objects
    page_ids = []
    for text in pages:
        lines = text.split("\n")
        ops = [b"BT /F1 11 Tf 14 TL 50 780 Td"]
        for line in lines:
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({esc}) Tj T*".encode("latin-1"))
        ops.append(b"ET")
        stream = b"\n".join(ops)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 842] /Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (pages_id, content, font)))
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages))) == pages_id
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    info = add(b"<< /Title (Test Handbook) /Author (QA) >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, info, xref))
    return out.getvalue()


def make_docx(blocks: List[tuple]) -> bytes:
    """blocks: ("h1"|"p", text) or ("table", [[cells], ...])."""
    import docx

    d = docx.Document()
    d.core_properties.title = "Policy Manual"
    for kind, value in blocks:
        if kind == "h1":
            d.add_heading(value, level=1)
        elif kind == "table":
            t = d.add_table(rows=len(value), cols=len(value[0]))
            for r, row in enumerate(value):
                for c, cell in enumerate(row):
                    t.cell(r, c).text = cell
        else:
            d.add_paragraph(value)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class FakeLLM:
    """Non-streaming fake: graph nodes fall back to ainvoke and emit the text as one token."""

    def __init__(self, reply: Callable[[list], str], species_name: str = "test-model"):
        self.species_name = species_name
        self._reply = reply
        self.calls: List[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self._reply(messages),
                         response_metadata={"fire": {"call_id": "call_x", "usage": {"input": 100, "output": 50}}})


class FakeStreamLLM(FakeLLM):
    """Streaming fake (same interface as FireChatModel.astream_chat)."""

    async def astream_chat(self, messages):
        self.calls.append(messages)
        text = self._reply(messages)
        for i in range(0, len(text), 12):
            yield "token", text[i:i + 12]
        yield "done", {"call_id": "call_s", "usage": {"input": 300, "output": 80}}


def last_user_text(messages) -> str:
    return str(messages[-1].content)
