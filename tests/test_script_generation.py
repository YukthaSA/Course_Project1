"""
Tests for assertion-based Playwright script generation and execution.
The integration tests drive a real browser against the built-in demo target
and are skipped automatically if Chromium is not available.
"""
import ast
import asyncio

import pytest

from config import config
from agents import ScriptGenerationAgent
from agents.script_builder import build_script
from engine.demo_app import start_demo_server
from engine.playwright_runner import PlaywrightRunner

VALID = {"email": "qa.user@example.com", "password": "ValidSecurePass123!"}


def _tc(i, title, data, assertions, ttype="Positive", tier=1):
    return {"id": f"TC-{i:03d}", "requirement_id": "REQ-001", "title": title, "tier": tier,
            "type": ttype, "test_data": data, "assertions": assertions}


def test_generated_script_is_valid_python_and_embeds_tier_labels():
    script = build_script([_tc(1, "Admin login", VALID, [{"type": "authenticated"}], tier=1),
                           _tc(2, "Profile page", {}, [{"type": "no_server_error"}], tier=3)],
                          "http://127.0.0.1:8080", True, 5000)
    ast.parse(script)
    assert script.index("[Tier 1] Admin login") < script.index("[Tier 3] Profile page")
    assert "JSON_RESULT" in script


def test_payload_with_quotes_survives_embedding():
    nasty = {"email": "' OR '1'='1' --", "password": "\"\"\"'''\\"}
    ast.parse(build_script([_tc(1, "inject", nasty, [{"type": "no_server_error"}])], "http://x", True, 5000))


def test_retry_widens_settle_wait(monkeypatch):
    agent = ScriptGenerationAgent()
    state = {"approved_test_cases": [_tc(1, "t", {}, [{"type": "no_server_error"}])], "target_url": "http://x",
             "retry_count": 2, "last_error": "Timeout", "error_analysis": {"root_cause": "Timeout"}}
    out = asyncio.run(agent.invoke(state))
    assert "SETTLE_MS = 1600" in out["current_script"]
    assert "Retry #2" in out["current_script"]


def test_runner_treats_skipped_as_neutral():
    runner = PlaywrightRunner(timeout=10000)
    mixed = runner.execute_script(
        "import json\nprint('JSON_RESULT: ' + json.dumps([{'test_case_id':'a','status':'PASSED'},"
        "{'test_case_id':'b','status':'SKIPPED'}]))", "t_mixed.py")
    assert mixed["status"] == "PASSED"
    all_skipped = runner.execute_script(
        "import json\nprint('JSON_RESULT: ' + json.dumps([{'test_case_id':'a','status':'SKIPPED'}]))", "t_skip.py")
    assert all_skipped["status"] == "FAILED_NO_TESTS"


@pytest.fixture(scope="module")
def demo_url():
    server, port = start_demo_server(0)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _browser_available():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _browser_available(), reason="Chromium not available")
def test_assertions_really_pass_and_fail_against_demo_target(demo_url):
    cases = [
        _tc(1, "valid login", VALID, [{"type": "authenticated"}, {"type": "no_server_error"}]),
        _tc(2, "wrong password rejected", {**VALID, "password": "nope"},
            [{"type": "error_shown", "value": ["invalid", "incorrect"]}, {"type": "not_authenticated"}], "Negative"),
        _tc(3, "malformed email rejected", {"email": "userexample.com", "password": "x"},
            [{"type": "error_shown", "value": ["email", "valid"]}], "Negative"),
        _tc(4, "empty fields blocked", {"email": "", "password": ""},
            [{"type": "error_shown", "value": ["required"]}, {"type": "not_authenticated"}], "Negative"),
        _tc(5, "xss does not execute", {"email": "<script>alert(1)</script>", "password": "x"},
            [{"type": "no_dialog"}, {"type": "no_server_error"}], "Security"),
        # control: deliberately wrong expectation - valid login must NOT be reported as an error
        _tc(6, "CONTROL wrong expectation", VALID, [{"type": "error_shown", "value": ["invalid"]}]),
        _tc(7, "no assertions", VALID, []),
        _tc(8, "idle timeout", {"idle_minutes": 31}, [{"type": "not_authenticated"}], "Negative"),
    ]
    script = build_script(cases, demo_url, True, config.TIMEOUT)
    result = PlaywrightRunner(timeout=config.TIMEOUT).execute_script(script, "t_integration.py")
    by_id = {r["test_case_id"]: r for r in result["test_results"]}

    assert [by_id[f"TC-00{i}"]["status"] for i in range(1, 9)] == \
        ["PASSED", "PASSED", "PASSED", "PASSED", "PASSED", "FAILED", "SKIPPED", "SKIPPED"]
    assert "error_shown" in by_id["TC-006"]["error"]
    assert result["status"] == "FAILED"          # the control failure makes the run fail
    assert all("duration" in r for r in result["test_results"])
