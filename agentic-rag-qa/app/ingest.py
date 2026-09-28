"""Load documents from disk into a vector store."""
from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_text
from app.store import VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt"}


def ingest_text(store: VectorStore, source: str, text: str, chunk_size: int, overlap: int) -> int:
    """(Re)index one document. Old chunks for the same source are replaced."""
    store.delete_source(source)
    chunks = chunk_text(text, source, chunk_size, overlap)
    store.add(chunks)
    return len(chunks)


def ingest_directory(store: VectorStore, docs_dir: Path, chunk_size: int, overlap: int) -> int:
    total = 0
    for path in sorted(Path(docs_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            total += ingest_text(store, path.name, path.read_text(encoding="utf-8"), chunk_size, overlap)
    return total
