"""
Test Design Agent - Converts validated requirements into structured test cases and BDD / Gherkin scenarios
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class TestDesignAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Test_Design", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Designing structured test cases and BDD / Gherkin specifications...")
        reqs = state.get("validated_requirements", state.get("requirements", []))
        if not reqs:
            self.log_error("No requirements found to design test cases")
            return state

        prompt = f"""Design comprehensive test cases based on these requirements:
{json.dumps(reqs[:15], indent=2)}

For each requirement, generate structured test cases including positive paths, negative paths, and security/boundary tests.
Provide a JSON array where each object has:
- id: "TC-001", "TC-002", etc.
- title: clear summary of test scenario
- type: "Positive", "Negative", "Security", or "Boundary"
- requirement_id: corresponding requirement ID
- preconditions: string or array of setup conditions
- steps: array of action steps to perform
- expected_result: expected system outcome
- gherkin: BDD Gherkin scenario string (Scenario, Given, When, Then)

Return ONLY valid JSON array:"""

        def _deterministic_test_designer() -> str:
            test_cases = []
            tc_num = 1
            for r in reqs[:10]:
                rid = r.get("id", f"REQ-{tc_num:03d}")
                desc = r.get("description", "System feature verification")
                cat = r.get("category", "Functional")

                # Positive test case
                pos_tc = {
                    "id": f"TC-{tc_num:03d}",
                    "title": f"Verify Successful {desc[:50]}",
                    "type": "Positive",
                    "requirement_id": rid,
                    "preconditions": "Application is accessible and in ready state",
                    "steps": [
                        "Navigate to application interface",
                        f"Submit valid inputs adhering to {rid}",
                        "Observe system confirmation and response"
                    ],
                    "expected_result": f"System completes action according to specification: {desc[:60]}",
                    "gherkin": f"Scenario: Successfully verify {rid}\n  Given the user is on the application home page\n  When valid parameters matching '{desc[:40]}' are entered\n  Then the system responds with success status"
                }
                test_cases.append(pos_tc)
                tc_num += 1

                # If Security or Functional, add negative / boundary test case
                if cat in ["Security", "Functional"] or "login" in desc.lower() or "password" in desc.lower():
                    neg_tc = {
                        "id": f"TC-{tc_num:03d}",
                        "title": f"Validate Negative Input Handling for {rid}",
                        "type": "Negative",
                        "requirement_id": rid,
                        "preconditions": "Application is running and reachable",
                        "steps": [
                            "Navigate to target input endpoint",
                            "Submit invalid credentials or malformed parameters",
                            "Verify rejection message and absence of unauthorized access"
                        ],
                        "expected_result": "System safely rejects invalid input with descriptive error message without crashing",
                        "gherkin": f"Scenario: Negative validation for {rid}\n  Given the application is accessible\n  When malformed or unauthorized inputs are supplied\n  Then an appropriate error indication is displayed and access is blocked"
                    }
                    test_cases.append(neg_tc)
                    tc_num += 1

            return json.dumps(test_cases, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Principal Software Test Engineer designing automated test suites.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_deterministic_test_designer
            )

            tc_list = self._parse_json_response(response_text)
            if not isinstance(tc_list, list) or len(tc_list) == 0:
                tc_list = json.loads(_deterministic_test_designer())

            state["test_cases"] = tc_list
            state["gherkin_scenarios"] = [tc.get("gherkin", "") for tc in tc_list if tc.get("gherkin")]
            if "metrics" in state and isinstance(state["metrics"], dict):
                state["metrics"]["total_test_cases"] = len(tc_list)

            self.log_success(f"Generated {len(tc_list)} test cases ({len(state['gherkin_scenarios'])} BDD scenarios)")
            return state
        except Exception as e:
            self.log_error(f"Error designing test cases: {e}")
            fallback = json.loads(_deterministic_test_designer())
            state["test_cases"] = fallback
            state["gherkin_scenarios"] = [tc.get("gherkin", "") for tc in fallback if tc.get("gherkin")]
            return state
