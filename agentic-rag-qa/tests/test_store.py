from app.chunking import Chunk
from app.ingest import ingest_text
from app.store import InMemoryStore


def test_search_finds_relevant_document(store):
    results = store.search("mileage reimbursement per kilometre", k=2)
    assert results
    assert results[0].source == "expense-policy.md"


def test_results_are_sorted_by_score(store):
    scores = [r.score for r in store.search("annual leave days", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_and_empty_query_return_nothing():
    s = InMemoryStore()
    assert s.search("anything", k=3) == []
    s.add([Chunk("a::0", "some text", "a")])
    assert s.search("the and of", k=3) == []  # stopwords only


def test_reingesting_a_source_replaces_old_chunks():
    s = InMemoryStore()
    ingest_text(s, "note.md", "old content about bananas", 600, 100)
    ingest_text(s, "note.md", "new content about apples", 600, 100)
    assert s.count() == 1
    assert s.search("bananas", k=1) == []
    assert s.search("apples", k=1)[0].source == "note.md"
