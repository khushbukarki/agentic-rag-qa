import pytest

from app.chunking import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello world.", "a.md", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].id == "a.md::0"
    assert chunks[0].source == "a.md"


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_text(text, "doc.md", chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_long_paragraph_is_hard_split():
    chunks = chunk_text("x" * 1000, "doc.md", chunk_size=300, overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks).count("x") >= 1000


def test_ids_are_unique():
    text = "\n\n".join("para " * 40 for _ in range(10))
    ids = [c.id for c in chunk_text(text, "doc.md", chunk_size=250, overlap=40)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", "doc.md", chunk_size=size, overlap=overlap)
