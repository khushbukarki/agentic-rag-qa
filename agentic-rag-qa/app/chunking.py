"""Split documents into overlapping chunks for retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str


def chunk_text(text: str, source: str, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Pack paragraphs into chunks of roughly `chunk_size` characters.

    Paragraph boundaries are kept where possible so a chunk rarely starts
    mid-sentence. A paragraph longer than `chunk_size` is hard-split. Each new
    chunk begins with the last `overlap` characters of the previous one so an
    answer that straddles a boundary can still be retrieved.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            step = chunk_size - overlap
            pieces.extend(para[i:i + chunk_size] for i in range(0, len(para), step))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= chunk_size else piece
        else:
            current = piece
    if current:
        chunks.append(current)

    return [Chunk(id=f"{source}::{i}", text=c, source=source) for i, c in enumerate(chunks)]
