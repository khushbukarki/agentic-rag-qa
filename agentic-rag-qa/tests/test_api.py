from fastapi.testclient import TestClient

from app.agent import LLMResponse
from app.config import Settings
from app.main import create_app
from app.store import InMemoryStore
from tests.conftest import DOCS_DIR, LOOKUP_PATH, ScriptedLLM


def make_client(llm=None) -> TestClient:
    settings = Settings(docs_dir=DOCS_DIR, lookup_path=LOOKUP_PATH, store_backend="memory")
    return TestClient(create_app(settings, store=InMemoryStore(), llm=llm, use_env_llm=False))


def test_health_reports_indexed_chunks():
    with make_client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["llm"] is False


def test_ask_extractive():
    with make_client() as client:
        res = client.post("/ask", json={"question": "What is the meal allowance for international travel?"})
    assert res.status_code == 200
    assert "$70" in res.json()["answer"]
    assert res.json()["mode"] == "extractive"


def test_ask_with_llm():
    llm = ScriptedLLM([LLMResponse(text="It's 20 days [leave-policy.md].")])
    with make_client(llm) as client:
        res = client.post("/ask", json={"question": "How much annual leave?"})
    assert res.json() == {"answer": "It's 20 days [leave-policy.md].", "sources": [],
                          "steps": [], "mode": "agent"}


def test_ask_validates_input():
    with make_client() as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422


def test_search_endpoint():
    with make_client() as client:
        hits = client.get("/search", params={"q": "stolen laptop", "k": 2}).json()
    assert hits[0]["source"] == "it-security-policy.md"


def test_upload_document_then_ask():
    with make_client() as client:
        doc = ("parking.md", b"Staff parking costs $5 per day.", "text/markdown")
        up = client.post("/documents", files={"file": doc})
        assert up.status_code == 201
        res = client.post("/ask", json={"question": "How much is staff parking?"})
    assert "$5 per day" in res.json()["answer"]


def test_upload_rejects_unsupported_types():
    with make_client() as client:
        res = client.post("/documents", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 415
