from app.agent import Agent, LLMResponse, ToolCall
from tests.conftest import ScriptedLLM


def test_agent_chains_search_and_calculate(store):
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("t1", "search_documents", {"query": "mileage rate"})]),
        LLMResponse(tool_calls=[ToolCall("t2", "calculate", {"expression": "120 * 0.95"})]),
        LLMResponse(text="You'd be reimbursed $114.00 [expense-policy.md]."),
    ])
    result = Agent(store, {}, llm).ask("How much do I get for driving 120 km?")

    assert result.mode == "agent"
    assert "114" in result.answer
    assert [s["tool"] for s in result.steps] == ["search_documents", "calculate"]
    assert "expense-policy.md" in result.sources

    # The calculator's output must be sent back to the model as a tool_result.
    last_turn = llm.calls[2][-1]
    assert last_turn["role"] == "user"
    assert last_turn["content"][0] == {"type": "tool_result", "tool_use_id": "t2", "content": "114.0"}


def test_agent_stops_at_step_limit(store):
    loop = [LLMResponse(tool_calls=[ToolCall(f"t{i}", "search_documents", {"query": "leave"})])
            for i in range(3)]
    result = Agent(store, {}, ScriptedLLM(loop), max_steps=3).ask("leave?")
    assert "step limit" in result.answer
    assert len(result.steps) == 3


def test_extractive_mode_without_llm(store):
    result = Agent(store, {}, llm=None).ask("How long must passwords be?")
    assert result.mode == "extractive"
    assert "14 characters" in result.answer
    assert result.sources[0] == "it-security-policy.md"
