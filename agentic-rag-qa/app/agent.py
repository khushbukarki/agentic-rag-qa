"""Tool-calling agent loop.

The model decides at each step whether to search, calculate, look up a term
or answer, instead of following a fixed retrieve-then-generate pipeline.
When no LLM is configured the agent falls back to extractive answers
(the top retrieved passages), which keeps the API usable offline and in CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.store import SearchResult, VectorStore
from app.tools import TOOL_SPECS, ToolRunner

SYSTEM_PROMPT = (
    "You answer questions about the company's documents. Always use search_documents "
    "before answering a policy question, use calculate for any arithmetic, and use "
    "lookup_term for unfamiliar acronyms. Answer only from tool results. If the documents "
    "don't contain the answer, say so. Keep answers short and cite sources in [brackets]."
)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    """Adapter from the Anthropic Messages API to the agent's LLM interface."""

    def __init__(self, client: Any, model: str, max_tokens: int = 1024) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages
        )
        out = LLMResponse()
        for block in resp.content:
            if block.type == "text":
                out.text += block.text
            elif block.type == "tool_use":
                out.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return out


def build_llm(api_key: str, model: str) -> LLM | None:
    """Return a Claude-backed LLM, or None (extractive mode) when no key is set."""
    if not api_key:
        return None
    import anthropic

    return AnthropicLLM(anthropic.Anthropic(api_key=api_key), model)


@dataclass
class AgentResult:
    answer: str
    sources: list[str]
    steps: list[dict[str, Any]]
    mode: str
    retrieved: list[SearchResult] = field(default_factory=list)


class Agent:
    def __init__(self, store: VectorStore, lookup: dict[str, str], llm: LLM | None,
                 top_k: int = 4, max_steps: int = 5) -> None:
        self.store = store
        self.lookup = lookup
        self.llm = llm
        self.top_k = top_k
        self.max_steps = max_steps

    def ask(self, question: str) -> AgentResult:
        if self.llm is None:
            return self._extractive(question)

        runner = ToolRunner(self.store, self.lookup, self.top_k)
        messages: list[dict] = [{"role": "user", "content": question}]
        steps: list[dict[str, Any]] = []

        for _ in range(self.max_steps):
            resp = self.llm.complete(SYSTEM_PROMPT, messages, TOOL_SPECS)
            if not resp.tool_calls:
                return self._result(resp.text.strip(), runner, steps, "agent")

            assistant_content: list[dict] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            assistant_content += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls
            ]
            messages.append({"role": "assistant", "content": assistant_content})

            results = []
            for call in resp.tool_calls:
                output = runner.run(call.name, call.input)
                steps.append({"tool": call.name, "input": call.input})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
            messages.append({"role": "user", "content": results})

        return self._result("I couldn't finish answering within the step limit.", runner, steps, "agent")

    def _extractive(self, question: str) -> AgentResult:
        runner = ToolRunner(self.store, self.lookup, self.top_k)
        runner.run("search_documents", {"query": question})
        top = runner.retrieved[:2]
        answer = (
            "\n\n".join(f"[{r.source}] {r.text}" for r in top)
            if top else "No relevant passages found in the documents."
        )
        steps = [{"tool": "search_documents", "input": {"query": question}}]
        return self._result(answer, runner, steps, "extractive")

    @staticmethod
    def _result(answer: str, runner: ToolRunner, steps: list, mode: str) -> AgentResult:
        sources = list(dict.fromkeys(r.source for r in runner.retrieved))
        return AgentResult(answer, sources, steps, mode, list(runner.retrieved))
