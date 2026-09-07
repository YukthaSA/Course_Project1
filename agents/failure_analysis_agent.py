"""
Failure Analysis Agent - Analyzes Playwright execution logs and errors to generate targeted fix recommendations
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class FailureAnalysisAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Failure_Analysis", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
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
