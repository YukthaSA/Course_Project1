"""
Failure Analysis Agent - Analyzes Playwright execution logs and errors to generate targeted fix recommendations
"""
import json
import re
from state import AgentState
from .base_agent import BaseAgent
from .test_design_agent import ASSERTION_TYPES
from typing import List, Dict, Any, Tuple

DEFECT = "APPLICATION_DEFECT"
SCRIPT = "SCRIPT_ISSUE"
ENV = "ENVIRONMENT_ISSUE"
SEVERITY_BY_TIER = {1: "High", 2: "Medium", 3: "Low"}

_ENV_RE = re.compile(r"net::|err_[a-z_]+|connection (refused|reset)|unreachable|dns|target (page|closed)|browser has been closed|crash", re.I)
_SCRIPT_RE = re.compile(r"timeout|waiting for|locator|selector|strict mode|not attached|detached|element is not|not visible|intercepts pointer", re.I)


def _failed_assertions(error: str) -> List[Tuple[str, str]]:
    """Parse 'kind: detail; kind: detail' (written by the generated script) into pairs."""
    found = []
    for part in (error or "").split("; "):
        kind, _, detail = part.partition(": ")
        if kind.strip() in ASSERTION_TYPES:
            found.append((kind.strip(), detail.strip()))
    return found


def _defect_note(kind: str, detail: str, data: Dict[str, Any]) -> str:
    shown = ", ".join(f"{k}={str(v)[:30]!r}" for k, v in (data or {}).items()) or "no input"
    if kind == "error_shown":
        if "no error" in detail:
            return (f"Input ({shown}) was accepted with no error or validation message. "
                    "Add input validation and show a visible message.")
        return (f"A message appeared but matched none of the expected keywords ({detail}). Either align the "
                "message wording or update the test's keywords - this one may be a test-side mismatch.")
    return {
        "no_error_shown": f"An error appeared for input that should be valid ({detail}). Check the validation rule.",
        "not_authenticated": f"Access was granted for input that should be rejected ({shown}). Treat as a security defect.",
        "authenticated": "Valid input did not reach a logged-in state. Check the success path.",
        "no_server_error": f"Input ({shown}) caused a server error. Handle it gracefully and return a validation error.",
        "no_dialog": "A JavaScript dialog fired for the payload - possible XSS. Escape or sanitise the output.",
    }.get(kind, f"Expected behaviour not met: {detail}")


def classify_result(result: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one FAILED test result as an application defect, script issue or environment issue."""
    err = str(result.get("error") or "")
    asserts = _failed_assertions(err)
    if asserts:
        category = DEFECT
        suggestion = " ".join(_defect_note(k, d, test_case.get("test_data")) for k, d in asserts[:2])
        likelihood = "Possible test-side keyword mismatch" if any(
            k == "error_shown" and "no error" not in d for k, d in asserts) else "Likely application defect"
    elif _ENV_RE.search(err) and not _SCRIPT_RE.search(err):
        category, likelihood = ENV, "Target or browser problem"
        suggestion = "The target app or browser was unreachable/closed. Check that the app is running and the URL is correct."
    else:
        category, likelihood = SCRIPT, "Script or locator problem"
        suggestion = "The test could not interact with the page (timeout or locator). Retrying with longer waits; if it persists, review the field locators."
    tier = result.get("tier", test_case.get("tier", 2))
    return {
        "test_case_id": result.get("test_case_id"),
        "requirement_id": result.get("requirement_id"),
        "title": result.get("title") or test_case.get("title"),
        "tier": tier,
        "severity": SEVERITY_BY_TIER.get(tier, "Medium") if category == DEFECT else "n/a",
        "category": category,
        "likelihood": likelihood,
        "error": err,
        "suggestion": suggestion,
    }


class FailureAnalysisAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Failure_Analysis", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        results = state.get("test_results") or []
        if any(r.get("status") == "FAILED" for r in results):
            return self._analyze_test_results(state, results)
        return await self._analyze_run_failure(state)

    def _analyze_test_results(self, state: AgentState, results: List[Dict[str, Any]]) -> AgentState:
        """Per-test analysis: separate real application defects from script/environment problems."""
        self.log_start("Classifying failed test cases (application defect vs script vs environment)...")
        cases = {tc.get("id"): tc for tc in state.get("test_cases", [])}
        failures = [classify_result(r, cases.get(r.get("test_case_id"), {}))
                    for r in results if r.get("status") == "FAILED"]
        summary = {
            "PASSED": sum(1 for r in results if r.get("status") == "PASSED"),
            "SKIPPED": sum(1 for r in results if r.get("status") == "SKIPPED"),
            DEFECT: sum(1 for f in failures if f["category"] == DEFECT),
            SCRIPT: sum(1 for f in failures if f["category"] == SCRIPT),
            ENV: sum(1 for f in failures if f["category"] == ENV),
        }
        if summary[SCRIPT]:
            root_cause, reason = "Script/Locator Issue", "script_issues"
            action = "Retry with longer settle waits; review field locators for the failing tests."
        elif summary[ENV]:
            root_cause, reason = "Network/Environment Error", "environment"
            action = "Start the target app or fix TARGET_URL, then re-run. Retrying the script will not help."
        else:
            root_cause, reason = "Application Defect (assertion mismatch)", "application_defects"
            action = "Do not retry: the tests ran correctly and the app behaved differently. Triage the defects below."
        analysis = {
            "root_cause": root_cause,
            "analysis_details": (f"{len(failures)} of {len(results)} tests failed: {summary[DEFECT]} application "
                                 f"defect(s), {summary[SCRIPT]} script issue(s), {summary[ENV]} environment issue(s); "
                                 f"{summary['SKIPPED']} skipped, {summary['PASSED']} passed."),
            "corrective_action": action,
            "patch_code": "",
            "can_auto_recover": summary[SCRIPT] > 0,
            "escalation_reason": reason,
            "summary": summary,
            "failures": failures,
        }
        state["error_analysis"] = analysis

        if "learning_memory" in state and isinstance(state["learning_memory"], dict):
            mem = state["learning_memory"].setdefault("common_failures", [])
            for f in failures[:10]:
                mem.append({"retry": state.get("retry_count", 0), "error": f["error"][:200],
                            "root_cause": f["category"], "fix_action": f["suggestion"][:200]})

        self.log_success(f"{analysis['analysis_details']} Auto-recoverable: {analysis['can_auto_recover']}")
        return state

    async def _analyze_run_failure(self, state: AgentState) -> AgentState:
        """Run-level analysis for crashes with no per-test results (original behaviour)."""
        self.log_start("Analyzing Playwright script execution failure...")
        exec_result = state.get("execution_result", {})
        last_error = state.get("last_error", exec_result.get("error", "Unknown execution error"))
        stdout = exec_result.get("stdout", "")
        stderr = exec_result.get("stderr", "")
        script_code = state.get("current_script", "")

        prompt = f"""Perform root cause analysis on the following Playwright script execution failure.

Error Message:
{last_error}

Console Stdout:
{stdout[:1000]}

Console Stderr:
{stderr[:1000]}

Current Script Code Snippet:
{script_code[:1500]}

Analyze the error and provide a JSON object response with:
1. "root_cause": Categorized cause (e.g. "Locator NotFound", "Timeout Waiting for Selector", "Assertion Mismatch", "Network/DNS Error")
2. "analysis_details": Concise technical explanation of why the script failed
3. "corrective_action": Concrete instruction for patching the Playwright script
4. "patch_code": Suggested replacement Python code block or locator fix
5. "can_auto_recover": boolean (true if automated retry with updated locator/wait can fix it, false if server is down/unreachable)

Return ONLY valid JSON:"""

        def _mock_failure_analysis() -> str:
            err_str = str(last_error).lower()
            if "timeout" in err_str or "waiting for" in err_str:
                rc = "Timeout Waiting for Selector"
                action = "Increase wait timeout and add explicit page.wait_for_selector call before clicking."
                patch = "await page.wait_for_selector('input, button', timeout=10000)"
                recoverable = True
            elif "locator" in err_str or "not found" in err_str:
                rc = "Locator NotFound"
                action = "Use broader text-based or role-based selector strategy (e.g. page.get_by_role or text match)."
                patch = "await page.get_by_role('button', name=re.compile('submit|login', re.I)).click()"
                recoverable = True
            elif "dns" in err_str or "net::" in err_str or "http" in err_str or "connect" in err_str:
                rc = "Network/DNS Connection Error"
                action = "Target web server is unreachable or offline. Verify URL and network connectivity."
                patch = "page.goto(target_url, wait_until='load')"
                recoverable = False
            else:
                rc = "Assertion Mismatch"
                action = "Refine assertion check to account for dynamic loading states."
                patch = "await expect(page).to_have_title(re.compile('.*'), timeout=5000)"
                recoverable = True

            return json.dumps({
                "root_cause": rc,
                "analysis_details": f"Execution halted due to: {last_error}",
                "corrective_action": action,
                "patch_code": patch,
                "can_auto_recover": recoverable
            }, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are an Expert Test Automation Debugger and Site Reliability Specialist.",
                model_name=self.config.GEMINI_PRO_MODEL,
                mock_fallback=_mock_failure_analysis
            )

            analysis_data = self._parse_json_response(response_text)
            if not isinstance(analysis_data, dict):
                analysis_data = json.loads(_mock_failure_analysis())

            state["error_analysis"] = analysis_data

            # Track in learning memory
            if "learning_memory" in state and isinstance(state["learning_memory"], dict):
                failures = state["learning_memory"].setdefault("common_failures", [])
                failures.append({
                    "retry": state.get("retry_count", 0),
                    "error": str(last_error)[:200],
                    "root_cause": analysis_data.get("root_cause"),
                    "fix_action": analysis_data.get("corrective_action")
                })

            self.log_success(f"Failure Analysis Complete. Root Cause: [{analysis_data.get('root_cause')}]. Auto-recoverable: {analysis_data.get('can_auto_recover')}")
            return state
        except Exception as e:
            self.log_error(f"Error during failure analysis: {e}")
            state["error_analysis"] = json.loads(_mock_failure_analysis())
            return state
