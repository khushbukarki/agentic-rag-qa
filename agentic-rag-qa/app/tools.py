"""Tools the agent can choose between: retrieval, calculation and lookup."""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from collections.abc import Callable
from typing import Any

from app.store import SearchResult, VectorStore

# --- calculator --------------------------------------------------------------

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> float:
    """Evaluate arithmetic without `eval`: only numbers and + - * / // % ** ( )."""
    if len(expression) > 200:
        raise ValueError("Expression too long")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and arithmetic operators are allowed")

    try:
        tree = ast.parse(expression.replace(",", ""), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression!r}") from exc
    return _eval(tree)


# --- lookup ------------------------------------------------------------------

def load_lookup(path: Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in data.items()}


# --- tool registry -----------------------------------------------------------

TOOL_SPECS = [
    {
        "name": "search_documents",
        "description": "Search the indexed company documents. Use for any question about "
                       "policies, rules, amounts or procedures. Returns the most relevant passages.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '120 * 0.95'. Use for any maths "
                       "instead of computing it yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "lookup_term",
        "description": "Look up the definition of an internal acronym or system name (e.g. MFA, ExpenseHub).",
        "input_schema": {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    },
]


class ToolRunner:
    """Executes tool calls and records which passages were retrieved."""

    def __init__(self, store: VectorStore, lookup: dict[str, str], top_k: int = 4) -> None:
        self.store = store
        self.lookup = lookup
        self.top_k = top_k
        self.retrieved: list[SearchResult] = []

    def run(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "search_documents":
                results = self.store.search(str(args.get("query", "")), self.top_k)
                self.retrieved.extend(r for r in results if r.id not in {x.id for x in self.retrieved})
                if not results:
                    return "No relevant passages found."
                return "\n\n".join(f"[{r.source}] {r.text}" for r in results)
            if name == "calculate":
                value = safe_calculate(str(args.get("expression", "")))
                return str(round(value, 6))
            if name == "lookup_term":
                term = str(args.get("term", "")).strip().lower()
                return self.lookup.get(term, f"No definition found for '{term}'.")
            return f"Unknown tool: {name}"
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return f"Tool error: {exc}"
