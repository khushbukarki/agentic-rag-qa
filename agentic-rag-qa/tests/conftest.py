import os
from pathlib import Path

import pytest

# Make sure importing app.main never tries to reach Chroma or the Claude API.
os.environ["STORE_BACKEND"] = "memory"
os.environ.pop("ANTHROPIC_API_KEY", None)

from app.agent import LLMResponse  # noqa: E402
from app.ingest import ingest_directory  # noqa: E402
from app.store import InMemoryStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
LOOKUP_PATH = ROOT / "data" / "lookup.json"


class ScriptedLLM:
    """Fake LLM that replays scripted responses and records what it was sent."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, system, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    ingest_directory(s, DOCS_DIR, chunk_size=600, overlap=100)
    return s
