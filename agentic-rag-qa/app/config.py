"""Application settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    docs_dir: Path = Path("data/docs")
    lookup_path: Path = Path("data/lookup.json")
    store_backend: str = "chroma"          # "chroma" or "memory"
    chroma_dir: str = "chroma_data"
    collection_name: str = "documents"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    max_agent_steps: int = 5
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-5"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            docs_dir=Path(os.getenv("DOCS_DIR", "data/docs")),
            lookup_path=Path(os.getenv("LOOKUP_PATH", "data/lookup.json")),
            store_backend=os.getenv("STORE_BACKEND", "chroma"),
            chroma_dir=os.getenv("CHROMA_DIR", "chroma_data"),
            collection_name=os.getenv("COLLECTION_NAME", "documents"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("TOP_K", "4")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "5")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        )
