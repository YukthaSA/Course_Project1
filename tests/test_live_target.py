"""
Tests for live-target handling: an unreachable target is reported (never replaced by the demo
app), the site probe inspects a real page, and every test case gets its own screenshots.
The browser tests are skipped automatically if Chromium is not available.
"""
import asyncio
import os
import socket

import pytest

from config import config
from agents.script_builder import build_script
from engine.demo_app import start_demo_server
from engine.playwright_runner import PlaywrightRunner
from pipeline import AgenticQAPipeline


def _browser_available():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


needs_browser = pytest.mark.skipif(not _browser_available(), reason="Chromium not available")


@pytest.fixture(scope="module")
def demo_url():
    server, port = start_demo_server(0)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _closed_port_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


@needs_browser
def test_unreachable_target_stops_the_run_and_does_not_fall_back_to_demo():
    state = asyncio.run(AgenticQAPipeline().run("The system shall allow login.", target_url=_closed_port_url()))
    assert state["execution_result"]["status"] == "TARGET_UNREACHABLE"
    assert state["final_results"]["status"] == "TARGET_UNREACHABLE"
    assert state["target_mode"] == "external"
    assert not state["test_cases"]                      # nothing was generated or run against another site
    assert state["human_review_required"] is True


@needs_browser
def test_probe_inspects_live_page_and_saves_screenshots(demo_url, tmp_path):
    health = PlaywrightRunner().probe_target(demo_url, str(tmp_path))
    assert health["reachable"] and health["status"] == 200
    assert {c["status"] for c in health["checks"]} <= {"PASS", "WARN", "FAIL"}
    assert 0 <= health["score"] <= 100
    for key in ("desktop", "mobile"):
        assert os.path.getsize(health["screenshots"][key]) > 1000
    assert health["inventory"]["inputs"], "DOM inventory should list the page's input fields"


@needs_browser
def test_each_test_case_gets_its_own_screenshots_in_the_run_folder(demo_url, tmp_path):
    cases = [
        {"id": "TC-001", "requirement_id": "REQ-001", "title": "valid login", "tier": 1, "type": "Positive",
         "test_data": {"email": "qa.user@example.com", "password": "ValidSecurePass123!"},
         "assertions": [{"type": "authenticated"}]},
        {"id": "TC-002", "requirement_id": "REQ-001", "title": "empty fields", "tier": 1, "type": "Negative",
         "test_data": {"email": "", "password": ""}, "assertions": [{"type": "error_shown", "value": ["required"]}]},
        {"id": "TC-003", "requirement_id": "REQ-002", "title": "not for this page", "tier": 2, "type": "Positive",
         "test_data": {}, "assertions": [], "applicable": False},
    ]
    script = build_script(cases, demo_url, True, config.TIMEOUT, shot_dir=str(tmp_path))
    result = PlaywrightRunner(timeout=config.TIMEOUT).execute_script(script, "t_shots.py", shot_dir=str(tmp_path))
    by_id = {r["test_case_id"]: r for r in result["test_results"]}

    assert [by_id[i]["status"] for i in ("TC-001", "TC-002", "TC-003")] == ["PASSED", "PASSED", "SKIPPED"]
    assert "not applicable" in by_id["TC-003"]["error"]
    for r in by_id.values():
        assert all(os.path.getsize(s["path"]) > 1000 for s in r["screenshots"])
        assert r["screenshots"][0]["label"] == "loaded" and r["screenshots"][-1]["label"] == "result"
    assert "filled" in [s["label"] for s in by_id["TC-001"]["screenshots"]]
    # all files live in this run's folder, nothing is borrowed from elsewhere
    assert {os.path.dirname(s["path"]) for r in by_id.values() for s in r["screenshots"]} == {str(tmp_path)}
    # distinct images per test: the login result and the empty-form result are different pages
    first = open(by_id["TC-001"]["screenshots"][-1]["path"], "rb").read()
    second = open(by_id["TC-002"]["screenshots"][-1]["path"], "rb").read()
    assert first != second


def test_script_fails_loudly_on_unreachable_target_instead_of_using_another_site(tmp_path):
    script = build_script([{"id": "TC-001", "requirement_id": "R", "title": "t", "tier": 1, "type": "Positive",
                            "test_data": {}, "assertions": [{"type": "no_server_error"}]}],
                          _closed_port_url(), True, 3000, shot_dir=str(tmp_path))
    assert "demo_app" not in script and "start_demo_server" not in script
