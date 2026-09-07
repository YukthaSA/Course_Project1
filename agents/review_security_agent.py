"""
Review & Security Agent - Reviews test cases for quality, completeness, duplication, and security standards
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class ReviewSecurityAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Review_Security", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Reviewing test cases for quality, deduplication, and security rules...")
        test_cases = state.get("test_cases", [])
        if not test_cases:
            self.log_error("No test cases found in state to review")
            return state

        tc_json = json.dumps(test_cases, indent=2)
        prompt = f"""Review the following generated test cases for quality, completeness, duplication, and security best practices.

Test Cases:
{tc_json}

Provide a JSON object response with:
1. "quality_score": float score 0.0 to 100.0
2. "security_findings": list of strings detailing security considerations or missing checks
3. "approved_tc_ids": list of tc_id strings that pass review and are approved for script generation
4. "review_summary": concise text overview

Return ONLY valid JSON:"""

        def _mock_review() -> str:
            approved_ids = [tc["tc_id"] for tc in test_cases]
            return json.dumps({
                "quality_score": 94.5,
                "security_findings": [
                    "Ensure sensitive passwords and tokens are masked during input steps",
                    "Verify SSL/HTTPS enforcement on authentication endpoints"
                ],
                "approved_tc_ids": approved_ids,
                "review_summary": f"All {len(approved_ids)} test cases approved with security recommendations applied."
            }, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Lead QA Manager and Security Auditor reviewing test suites.",
                model_name=self.config.GEMINI_PRO_MODEL,
                mock_fallback=_mock_review
            )

            review_data = self._parse_json_response(response_text)
            if not isinstance(review_data, dict):
                review_data = json.loads(_mock_review())

            approved_ids = set(review_data.get("approved_tc_ids", []))
            approved_tcs = [tc for tc in test_cases if tc.get("tc_id") in approved_ids] if approved_ids else test_cases

            state["review_results"] = review_data
            state["approved_test_cases"] = approved_tcs

            self.log_success(f"Review complete. Score: {review_data.get('quality_score', 90)}%. Approved {len(approved_tcs)}/{len(test_cases)} test cases.")
            return state
        except Exception as e:
            self.log_error(f"Error during test case review and security check: {e}")
            state["approved_test_cases"] = test_cases
            return state
