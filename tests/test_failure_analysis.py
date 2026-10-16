"""
Tests for per-test failure analysis and the pipeline's retry/escalation decisions.
"""
import asyncio

from agents import FailureAnalysisAgent
from agents.failure_analysis_agent import classify_result, DEFECT, SCRIPT, ENV
from pipeline import AgenticQAPipeline

SRS = "The user shall log in using email and password.\nThe user shall search the product catalog by search text."


def _res(i, status, error=None, tier=1):
    return {"test_case_id": f"TC-{i:03d}", "requirement_id": "REQ-001", "title": f"t{i}", "tier": tier,
            "status": status, "error": error, "duration": 0.5}


def _analyze(results, cases=None):
    state = {"test_results": results, "test_cases": cases or [], "retry_count": 0, "learning_memory": {}}
    return asyncio.run(FailureAnalysisAgent().invoke(state))["error_analysis"]


def test_assertion_failures_are_application_defects_and_not_retried():
    a = _analyze([_res(1, "PASSED"),
                  _res(2, "FAILED", "not_authenticated: access was granted", tier=1),
                  _res(3, "FAILED", "error_shown: no error or validation message was shown", tier=3),
                  _res(4, "SKIPPED", "No file upload control")])
    assert a["summary"][DEFECT] == 2 and a["summary"]["SKIPPED"] == 1 and a["summary"]["PASSED"] == 1
    assert a["can_auto_recover"] is False
    assert a["escalation_reason"] == "application_defects"
    sev = {f["test_case_id"]: f["severity"] for f in a["failures"]}
    assert sev == {"TC-002": "High", "TC-003": "Low"}
    assert "security defect" in a["failures"][0]["suggestion"]


def test_wording_mismatch_is_flagged_as_possible_test_side_issue():
    f = classify_result(_res(1, "FAILED", "error_shown: message ['Oops'] contains none of ['invalid']"), {})
    assert f["category"] == DEFECT and "test-side" in f["likelihood"]


def test_timeouts_are_script_issues_and_recoverable():
    a = _analyze([_res(1, "FAILED", "TimeoutError: Page.fill: Timeout 5000ms exceeded waiting for locator"),
                  _res(2, "FAILED", "error_shown: no error or validation message was shown")])
    assert a["summary"][SCRIPT] == 1 and a["summary"][DEFECT] == 1
    assert a["can_auto_recover"] is True
    assert a["escalation_reason"] == "script_issues"


def test_unreachable_target_is_environment_and_not_recoverable():
    a = _analyze([_res(1, "FAILED", "Error: net::ERR_CONNECTION_REFUSED at http://x")])
    assert a["summary"][ENV] == 1 and a["can_auto_recover"] is False
    assert a["escalation_reason"] == "environment"


def test_run_level_crash_still_uses_original_analysis():
    state = {"last_error": "Timeout 5000ms exceeded waiting for locator('#submit')", "execution_result": {}}
    a = asyncio.run(FailureAnalysisAgent().invoke(state))["error_analysis"]
    assert a["root_cause"] == "Timeout Waiting for Selector" and a["can_auto_recover"] is True


class _FakeRunner:
    """Returns scripted execution results, one per attempt."""
    def __init__(self, *results):
        self.results, self.calls = list(results), 0

    def execute_script(self, script, script_name=""):
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


def _exec(status, test_results):
    return {"status": status, "error": f"{sum(r['status'] == 'FAILED' for r in test_results)} failed",
            "stdout": "", "stderr": "", "duration_seconds": 1, "screenshots": [], "test_results": test_results}


def _run(runner, max_retries=3):
    pipeline = AgenticQAPipeline()
    pipeline.runner = runner
    return asyncio.run(pipeline.run(srs_text=SRS, target_url="http://x", max_retries=max_retries))


def test_pipeline_stops_retrying_when_failures_are_real_defects():
    defect = _exec("FAILED", [_res(1, "FAILED", "not_authenticated: access was granted")])
    runner = _FakeRunner(defect)
    state = _run(runner)
    assert runner.calls == 1                                   # no pointless re-runs
    assert state["human_review_required"] is True
    ctx = state["human_review_context"]
    assert ctx["reason"] == "application_defects" and ctx["failures"][0]["category"] == DEFECT
    assert state["final_results"]["status"] == "REQUIRES_HUMAN_REVIEW"


def test_pipeline_retries_script_issues_and_recovers():
    broken = _exec("FAILED", [_res(1, "FAILED", "TimeoutError: waiting for locator")])
    fixed = _exec("PASSED", [_res(1, "PASSED")])
    runner = _FakeRunner(broken, fixed)
    state = _run(runner)
    assert runner.calls == 2
    assert state["human_review_required"] is False
    assert state["retry_history"][0]["root_cause"] == "Script/Locator Issue"
    assert "SETTLE_MS = 1000" in state["current_script"]       # retry widened the wait


def test_pipeline_escalates_after_max_retries_for_persistent_script_issues():
    broken = _exec("FAILED", [_res(1, "FAILED", "TimeoutError: waiting for locator")])
    runner = _FakeRunner(broken)
    state = _run(runner, max_retries=2)
    assert runner.calls == 2 and state["human_review_required"] is True
    assert state["human_review_context"]["reason"] == "script_issues"
