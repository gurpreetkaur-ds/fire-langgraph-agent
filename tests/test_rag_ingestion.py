"""Upload validation, text extraction, cleaning and chunking."""
from __future__ import annotations

import io
import zipfile

import pytest

from app.rag.chunking import SECTION_JOINER, chunk_sections, chunk_text
from app.rag.extract import ExtractionError, Section, UploadRejected, clean_text, detect_type, extract, safe_filename
from tests.rag_helpers import make_docx, make_pdf


# ---- validation ----

@pytest.mark.parametrize("name,data,expected", [
    ("a.pdf", make_pdf(["hello"]), "pdf"),
    ("a.DOCX", make_docx([("p", "hi")]), "docx"),
    ("a.txt", b"plain text", "txt"),
    ("a.md", b"# Title\nbody", "md"),
    ("a.markdown", b"# Title", "md"),
    ("a.csv", b"a,b\n1,2", "csv"),
])
def test_detect_type_accepts_real_files(name, data, expected):
    assert detect_type(name, data) == expected


@pytest.mark.parametrize("name,data,msg", [
    ("a.exe", b"MZ...", "Unsupported"),
    ("a", b"text", "Unsupported"),
    ("a.pdf", b"not a pdf at all", "does not look like a PDF"),
    ("a.docx", b"%PDF-1.4 fake", "does not look like a DOCX"),
    ("a.txt", b"\x00\x01\x02binary", "binary"),
    ("a.txt", b"", "empty"),
])
def test_detect_type_rejects_bad_files(name, data, msg):
    with pytest.raises(UploadRejected, match=msg):
        detect_type(name, data)


def test_docx_that_is_just_a_zip_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(UploadRejected, match="not a Word"):
        detect_type("x.docx", buf.getvalue())


def test_safe_filename_strips_paths_and_control_chars():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Users\\me\\report<1>.pdf") == "report1.pdf"
    assert safe_filename("") == "document"


# ---- extraction ----

def test_pdf_extraction_keeps_page_numbers_and_metadata(tmp_path):
    p = tmp_path / "h.pdf"
    p.write_bytes(make_pdf(["Vacation policy: employees get 25 days.", "Remote work is allowed on Fridays."]))
    doc = extract(p, "pdf")
    assert doc.page_count == 2
    assert [s.page for s in doc.sections] == [1, 2]
    assert "25 days" in doc.sections[0].text and "Fridays" in doc.sections[1].text
    assert doc.meta["title"] == "Test Handbook"


def test_pdf_with_no_text_fails_with_a_clear_message(tmp_path):
    p = tmp_path / "blank.pdf"
    p.write_bytes(make_pdf([""]))
    with pytest.raises(ExtractionError, match="scanned PDF"):
        extract(p, "pdf")


def test_pdf_page_limit(tmp_path):
    p = tmp_path / "long.pdf"
    p.write_bytes(make_pdf(["x"] * 3))
    with pytest.raises(ExtractionError, match="limit is 2"):
        extract(p, "pdf", max_pdf_pages=2)


def test_docx_extraction_keeps_headings_and_tables(tmp_path):
    p = tmp_path / "m.docx"
    p.write_bytes(make_docx([("h1", "Security"), ("p", "Passwords rotate every 90 days."),
                             ("table", [["Role", "Access"], ["Admin", "Full"]]), ("h1", "Travel"), ("p", "Book economy class.")]))
    doc = extract(p, "docx")
    assert [s.heading for s in doc.sections] == ["Security", "Travel"]
    assert "Admin | Full" in doc.sections[0].text
    assert doc.meta["title"] == "Policy Manual"


def test_markdown_sections_follow_headings_but_not_inside_code(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("# Setup\nRun install.\n```\n# not a heading\n```\n## Deploy\nUse systemd.\n")
    doc = extract(p, "md")
    assert [s.heading for s in doc.sections] == ["Setup", "Deploy"]
    assert "# not a heading" in doc.sections[0].text


def test_csv_rows_become_labelled_records(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("product,price,stock\nWidget,9.99,12\nGadget,24.50,0\n")
    doc = extract(p, "csv")
    assert doc.meta["columns"] == ["product", "price", "stock"] and doc.meta["row_count"] == 2
    assert "Row 2: product: Gadget; price: 24.50; stock: 0" in doc.sections[0].text
    assert doc.sections[0].heading == "Rows 1-2"


def test_txt_latin1_fallback(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes("café résumé".encode("cp1252"))
    assert extract(p, "txt").sections[0].text == "café résumé"


def test_clean_text_rejoins_hyphenation_and_collapses_whitespace():
    assert clean_text("exam-\nple   text\r\n\n\n\nnext\x07") == "example text\n\nnext"


# ---- chunking ----

def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    spans = chunk_text(text, 300, 60)
    assert len(spans) > 5
    assert all(e - s <= 300 for s, e in spans)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 < e1, "consecutive chunks overlap"
        assert e1 - s2 <= 60 + 40  # overlap close to the configured size (word-aligned)
    covered = set()
    for s, e in spans:
        covered.update(range(s, e))
    assert covered >= set(i for i, ch in enumerate(text) if not ch.isspace())  # nothing lost


def test_chunks_prefer_paragraph_boundaries():
    text = "Alpha paragraph one.\n\n" + "Beta paragraph two is here.\n\n" + "Gamma three."
    spans = chunk_text(text, 40, 0)
    assert [text[s:e].strip() for s, e in spans] == ["Alpha paragraph one.", "Beta paragraph two is here.", "Gamma three."]


def test_hard_cut_when_no_separator():
    spans = chunk_text("x" * 1000, 300, 0)
    assert [e - s for s, e in spans] == [300, 300, 300, 100]


def test_chunk_sections_keeps_provenance_and_offsets():
    sections = [Section("Page one text " * 40, page=1), Section("Second page words " * 40, page=2, heading="Intro")]
    chunks = chunk_sections(sections, 200, 40)
    full = SECTION_JOINER.join(s.text for s in sections)
    assert {c.page for c in chunks} == {1, 2}
    assert all(full[c.char_start:c.char_end] == c.text for c in chunks)
    assert all(c.section == "Intro" for c in chunks if c.page == 2)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_csv_header_detected_when_sniffer_is_fooled_by_mixed_columns(tmp_path):
    p = tmp_path / "pricing.csv"
    p.write_text("plan,monthly_price_eur,seats\nStarter,19,3\nTeam,79,10\nEnterprise,custom,unlimited\n")
    doc = extract(p, "csv")
    assert doc.meta["columns"] == ["plan", "monthly_price_eur", "seats"] and doc.meta["row_count"] == 3
    assert "Row 2: plan: Team; monthly_price_eur: 79; seats: 10" in doc.sections[0].text


def test_embedding_text_has_a_contextual_header():
    from app.rag.chunking import ChunkData
    from app.rag.ingest import embedding_text

    c = ChunkData(index=3, text="Q4 priorities: migrate search.", page=4, section=None, char_start=0, char_end=10)
    assert embedding_text("report.pdf", "Engineering Report", c) == "report.pdf — Engineering Report — page 4\nQ4 priorities: migrate search."
    md = ChunkData(index=0, text="Body", page=None, section="Setup", char_start=0, char_end=4)
    assert embedding_text("notes.md", None, md) == "notes.md — Setup\nBody"
