"""FastAPI entry point."""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agent import LLM, Agent, build_llm
from app.config import Settings
from app.ingest import SUPPORTED_SUFFIXES, ingest_directory, ingest_text
from app.store import VectorStore, build_store
from app.tools import load_lookup

MAX_UPLOAD_BYTES = 1_000_000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Step(BaseModel):
    tool: str
    input: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    steps: list[Step]
    mode: str


class SearchHit(BaseModel):
    source: str
    score: float
    text: str


def create_app(settings: Settings | None = None, store: VectorStore | None = None,
               llm: LLM | None = None, use_env_llm: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or build_store(settings.store_backend, settings.chroma_dir, settings.collection_name)
    if llm is None and use_env_llm:
        llm = build_llm(settings.anthropic_api_key, settings.model)
    agent = Agent(store, load_lookup(settings.lookup_path), llm, settings.top_k, settings.max_agent_steps)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if store.count() == 0 and settings.docs_dir.exists():
            ingest_directory(store, settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
        yield

    app = FastAPI(title="Agentic RAG Document Q&A", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": store.count(), "llm": llm is not None}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        result = agent.ask(req.question)
        return AskResponse(answer=result.answer, sources=result.sources,
                           steps=[Step(**s) for s in result.steps], mode=result.mode)

    @app.get("/search", response_model=list[SearchHit])
    def search(q: str, k: int = 4) -> list[SearchHit]:
        if not q.strip():
            raise HTTPException(400, "Query must not be empty")
        k = max(1, min(k, 20))
        return [SearchHit(source=r.source, score=r.score, text=r.text) for r in store.search(q, k)]

    @app.post("/documents", status_code=201)
    async def upload(file: Annotated[UploadFile, File()]) -> dict:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "")
        if not any(name.lower().endswith(s) for s in SUPPORTED_SUFFIXES):
            raise HTTPException(415, "Only .md and .txt files are supported")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 1 MB)")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 text") from None
        n = ingest_text(store, name, text, settings.chunk_size, settings.chunk_overlap)
        return {"source": name, "chunks": n}

    @app.post("/reindex")
    def reindex() -> dict:
        n = ingest_directory(store, settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
        return {"chunks_indexed": n}

    return app


app = create_app()
