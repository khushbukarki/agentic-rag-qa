import pytest

from app.tools import ToolRunner, load_lookup, safe_calculate
from tests.conftest import LOOKUP_PATH


@pytest.mark.parametrize("expr,expected", [
    ("120 * 0.95", 114.0),
    ("(2 + 3) * 4", 20),
    ("-5 + 10 / 4", -2.5),
    ("2 ** 10", 1024),
    ("1,000 + 1", 1001),
])
def test_calculator(expr, expected):
    assert safe_calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "open('secrets.txt')",
    "x + 1",
    "2 ** 1000",
    "1 +",
])
def test_calculator_rejects_unsafe_or_invalid_input(expr):
    with pytest.raises(ValueError):
        safe_calculate(expr)


def test_runner_returns_errors_instead_of_raising(store):
    runner = ToolRunner(store, {})
    assert runner.run("calculate", {"expression": "1/0"}).startswith("Tool error")
    assert runner.run("does_not_exist", {}).startswith("Unknown tool")


def test_search_tool_records_retrieved_chunks(store):
    runner = ToolRunner(store, {}, top_k=2)
    output = runner.run("search_documents", {"query": "password length"})
    assert "it-security-policy.md" in output
    assert runner.retrieved and runner.retrieved[0].source == "it-security-policy.md"


def test_lookup_is_case_insensitive(store):
    runner = ToolRunner(store, load_lookup(LOOKUP_PATH))
    assert "Multi-factor" in runner.run("lookup_term", {"term": "mfa"})
    assert "No definition" in runner.run("lookup_term", {"term": "XYZ"})
