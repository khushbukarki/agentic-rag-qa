import pytest

from eval.metrics import f1, groundedness, keyword_hit, precision_at_k, recall_at_k


def test_precision_and_recall():
    retrieved = ["a.md", "b.md", "a.md", "c.md"]
    assert precision_at_k(retrieved, {"a.md"}, 4) == 0.5
    assert recall_at_k(retrieved, {"a.md", "d.md"}, 4) == 0.5
    assert recall_at_k(retrieved, {"c.md"}, 2) == 0.0
    assert precision_at_k([], {"a.md"}, 3) == 0.0


def test_f1():
    assert f1(0.5, 0.5) == 0.5
    assert f1(0.0, 0.0) == 0.0


def test_groundedness_flags_unsupported_facts():
    context = ["Staff are reimbursed at $0.95 per kilometre for business travel."]
    grounded = "Staff are reimbursed $0.95 per kilometre [expense-policy.md]."
    invented = "Staff also receive a free parking pass every Tuesday."
    assert groundedness(grounded, context) == 1.0
    assert groundedness(f"{grounded} {invented}", context) == pytest.approx(0.5)
    assert groundedness("", context) == 0.0


def test_keyword_hit_is_case_insensitive():
    assert keyword_hit("Ask the finance director", ["Finance Director"])
    assert not keyword_hit("Ask your manager", ["Finance Director", "manager"])
