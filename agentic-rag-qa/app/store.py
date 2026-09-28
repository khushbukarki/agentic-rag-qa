"""Vector stores.

`ChromaStore` is used in production (semantic embeddings, persisted to disk).
`InMemoryStore` is a dependency-free TF-IDF store used by the test suite and
as a keyword-search baseline in the evaluation harness.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from app.chunking import Chunk
from app.text import tokenize


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    source: str
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def delete_source(self, source: str) -> None: ...
    def search(self, query: str, k: int) -> list[SearchResult]: ...
    def count(self) -> int: ...


class InMemoryStore:
    """TF-IDF cosine similarity over chunks held in memory."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = Counter(tokenize(chunk.text))

    def delete_source(self, source: str) -> None:
        for cid in [cid for cid, c in self._chunks.items() if c.source == source]:
            del self._chunks[cid]
            del self._tf[cid]

    def count(self) -> int:
        return len(self._chunks)

    def _idf(self) -> dict[str, float]:
        n = len(self._tf)
        df: Counter = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        return {term: math.log((1 + n) / (1 + d)) + 1 for term, d in df.items()}

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not self._chunks or k <= 0:
            return []
        idf = self._idf()
        q_vec = {t: c * idf.get(t, 0.0) for t, c in Counter(tokenize(query)).items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        scored = []
        for cid, tf in self._tf.items():
            d_vec = {t: c * idf[t] for t, c in tf.items()}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            if d_norm == 0:
                continue
            dot = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, cid))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            SearchResult(cid, self._chunks[cid].text, self._chunks[cid].source, round(score, 4))
            for score, cid in scored[:k]
        ]


class ChromaStore:
    """Persistent semantic search backed by Chroma's default embedding model."""

    def __init__(self, persist_dir: str | None = None, collection: str = "documents") -> None:
        import chromadb  # imported lazily so tests don't need it

        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, k: int) -> list[SearchResult]:
        total = self.count()
        if total == 0 or k <= 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=min(k, total))
        return [
            SearchResult(cid, doc, meta["source"], round(1 - dist, 4))  # cosine distance -> similarity
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
        ]


def build_store(backend: str, persist_dir: str | None = None, collection: str = "documents") -> VectorStore:
    if backend == "memory":
        return InMemoryStore()
    if backend == "chroma":
        return ChromaStore(persist_dir, collection)
    raise ValueError(f"Unknown store backend: {backend!r}")
