"""
Tests for per-stage LLM routing: which model a stage is assigned, the fallback chain when a
provider is missing or failing, and the trace that records which model really answered.
No network calls: provider adapters are replaced with fakes.
"""
import asyncio

import pytest

from config import config
from agents import base_agent
from agents.base_agent import BaseAgent
from agents.llm_providers import PROVIDERS


class Stage(BaseAgent):
    def __init__(self, name="Test_Design"):
        super().__init__(name)

    async def invoke(self, state):
        return state


@pytest.fixture(autouse=True)
def restore_config():
    saved = (dict(config.STAGE_ROUTES), config.LLM_PROVIDER)
    yield
    config.STAGE_ROUTES.clear()
    config.STAGE_ROUTES.update(saved[0])
    config.LLM_PROVIDER = saved[1]


def fake_providers(monkeypatch, available, behaviour):
    """available: providers that count as configured; behaviour: provider -> text or Exception."""
    calls = []

    def make(name):
        def adapter(cfg, model, system, prompt):
            calls.append((name, model))
            outcome = behaviour[name]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return adapter

    monkeypatch.setattr(base_agent, "PROVIDERS", {p: make(p) for p in ("gemini", "groq", "openai", "anthropic", "ollama")})
    monkeypatch.setattr(config, "provider_available", lambda p: p in available)
    return calls


def test_every_stage_has_its_own_route_and_models_differ():
    routes = {s: config.get_stage_route(s) for s in config.STAGE_ORDER}
    assert set(routes) == set(config.STAGE_ORDER) and len(routes) == 6
    assert len({r for r in routes.values()}) > 3          # not one model for everything
    assert all(p in PROVIDERS for p, _ in routes.values())


def test_env_override_changes_one_stage(monkeypatch):
    monkeypatch.setenv("LLM_ROUTE_TEST_DESIGN", "groq:openai/gpt-oss-120b")
    from config import Config
    assert Config().STAGE_ROUTES["Test_Design"] == ("groq", "openai/gpt-oss-120b")
    assert Config().STAGE_ROUTES["Requirement_Extraction"] == config.STAGE_ROUTE_DEFAULTS["Requirement_Extraction"]


def test_stage_uses_its_assigned_provider_and_model(monkeypatch):
    config.LLM_PROVIDER = "auto"
    config.STAGE_ROUTES["Test_Design"] = ("groq", "openai/gpt-oss-120b")
    calls = fake_providers(monkeypatch, {"gemini", "groq"}, {"groq": "from-groq", "gemini": "from-gemini"})
    agent = Stage("Test_Design")
    assert asyncio.run(agent.call_llm("p")) == "from-groq"
    assert calls == [("groq", "openai/gpt-oss-120b")]
    assert agent.call_log[0]["provider"] == "groq" and agent.call_log[0]["fallback"] is False


def test_different_stages_reach_different_providers(monkeypatch):
    config.LLM_PROVIDER = "auto"
    config.STAGE_ROUTES["Requirement_Extraction"] = ("gemini", "gemini-3.1-flash-lite")
    config.STAGE_ROUTES["Requirement_Validation"] = ("groq", "openai/gpt-oss-20b")
    calls = fake_providers(monkeypatch, {"gemini", "groq"}, {"gemini": "g", "groq": "q"})
    asyncio.run(Stage("Requirement_Extraction").call_llm("p"))
    asyncio.run(Stage("Requirement_Validation").call_llm("p"))
    assert calls == [("gemini", "gemini-3.1-flash-lite"), ("groq", "openai/gpt-oss-20b")]


def test_failing_provider_falls_back_to_next_available_and_is_traced(monkeypatch):
    config.LLM_PROVIDER = "auto"
    config.STAGE_ROUTES["Test_Design"] = ("openai", "gpt-4o")
    calls = fake_providers(monkeypatch, {"openai", "gemini", "groq"},
                           {"openai": RuntimeError("HTTP 429 no credits"), "gemini": "from-gemini", "groq": "from-groq"})
    agent = Stage("Test_Design")
    assert asyncio.run(agent.call_llm("p")) == "from-gemini"
    assert [c[0] for c in calls] == ["openai", "gemini"]
    rec = agent.call_log[0]
    assert rec["fallback"] is True and rec["planned"] == "openai/gpt-4o" and "429" in rec["errors"][0]


def test_unconfigured_primary_is_skipped_without_a_call(monkeypatch):
    config.LLM_PROVIDER = "auto"
    config.STAGE_ROUTES["Test_Design"] = ("anthropic", "claude")
    calls = fake_providers(monkeypatch, {"groq"}, {"groq": "from-groq"})
    assert asyncio.run(Stage("Test_Design").call_llm("p")) == "from-groq"
    assert [c[0] for c in calls] == ["groq"]


def test_all_providers_down_uses_deterministic_fallback(monkeypatch):
    config.LLM_PROVIDER = "auto"
    fake_providers(monkeypatch, set(), {})
    agent = Stage("Test_Design")
    assert asyncio.run(agent.call_llm("p", mock_fallback=lambda: "rule-based")) == "rule-based"
    assert agent.call_log[0]["provider"] == "deterministic"


def test_no_provider_and_no_fallback_raises(monkeypatch):
    config.LLM_PROVIDER = "auto"
    fake_providers(monkeypatch, set(), {})
    with pytest.raises(RuntimeError):
        asyncio.run(Stage("Test_Design").call_llm("p"))


def test_forced_single_engine_never_silently_uses_another_provider(monkeypatch):
    config.LLM_PROVIDER = "groq"
    calls = fake_providers(monkeypatch, {"groq", "gemini"}, {"groq": RuntimeError("down"), "gemini": "from-gemini"})
    agent = Stage("Test_Design")
    assert asyncio.run(agent.call_llm("p", mock_fallback=lambda: "rule-based")) == "rule-based"
    assert [c[0] for c in calls] == ["groq"]


def test_pipeline_trace_reports_assigned_and_answering_model(monkeypatch):
    from pipeline import AgenticQAPipeline
    config.LLM_PROVIDER = "auto"
    fake_providers(monkeypatch, {"gemini"}, {"gemini": "ok"})
    pipe = AgenticQAPipeline()
    asyncio.run(pipe.req_extractor.call_llm("p"))
    rows = {r["stage"]: r for r in pipe._llm_trace()}
    assert rows["Requirement_Extraction"]["answered_by"].startswith("gemini/")
    assert rows["Failure_Analysis"]["answered_by"] == "not needed"
