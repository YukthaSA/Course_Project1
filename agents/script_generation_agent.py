"""
Script Generation Agent - Converts approved test cases into an executable Playwright Python script.

The script is built deterministically from each test case's test_data and assertions (see
script_builder.py), so every case becomes a real pass/fail check instead of free-form,
unverifiable LLM code. On retries it widens the settle wait and records the failure context.
"""
from state import AgentState
from .base_agent import BaseAgent
from .script_builder import build_script


class ScriptGenerationAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Script_Generation", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Generating assertion-based Playwright test script...")
        approved_tcs = state.get("approved_test_cases", state.get("test_cases", []))
        target_url = state.get("target_url", self.config.TARGET_URL)
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error", None)
        error_analysis = state.get("error_analysis", None)

        if not approved_tcs:
            self.log_error("No approved test cases available for script generation")
            return state

        # Tier 1 runs first (stable sort keeps requirement order inside a tier)
        approved_tcs = sorted(approved_tcs, key=lambda tc: tc.get("tier", 2))

        retry_note = ""
        if retry_count > 0 and last_error:
            self.log_info(f"Regenerating script for Retry #{retry_count} with longer settle waits...")
            cause = (error_analysis or {}).get("root_cause", "unknown")
            retry_note = f"Retry #{retry_count} - previous failure ({cause}): {str(last_error)[:120]}".replace('"', "'")

        script_code = build_script(
            approved_tcs,
            target_url=target_url,
            headless=self.config.HEADLESS,
            timeout_ms=self.config.TIMEOUT,
            settle_ms=400 + 600 * retry_count,
            retry_note=retry_note,
        )

        state["current_script"] = script_code
        state.setdefault("all_scripts", {})[f"v{retry_count + 1}"] = script_code
        checked = sum(1 for tc in approved_tcs if tc.get("assertions"))
        self.log_success(f"Generated Playwright script ({len(script_code.splitlines())} lines, "
                         f"{len(approved_tcs)} tests, {checked} with assertions)")
        return state
