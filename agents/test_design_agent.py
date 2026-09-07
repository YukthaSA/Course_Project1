"""
Test Design Agent - Transforms requirements into structured functional, negative, edge, and security test cases
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class TestDesignAgent(BaseAgent):
    __test__ = False  # Instruct pytest to ignore class collection

    def __init__(self, model_client=None):
        super().__init__("Test_Design", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Generating structured test cases from validated requirements...")
        requirements = state.get("validated_requirements", state.get("requirements", []))
        testing_priorities = state.get("testing_priorities", [])
        target_url = state.get("target_url", self.config.TARGET_URL)

        if not requirements:
            self.log_error("No requirements found to design test cases")
            return state

        req_input = json.dumps(requirements, indent=2)
        prio_input = json.dumps(testing_priorities, indent=2)

        prompt = f"""Generate detailed end-to-end test cases for each requirement.
Target Application URL: {target_url}

Requirements:
{req_input}

Testing Priorities:
{prio_input}

For each requirement, create at least 1-2 structured test cases covering:
1. Positive Functional Scenarios
2. Negative / Edge Scenarios
3. Basic Security / Validation Scenarios (if applicable)

Return ONLY a JSON array of test case objects with fields:
- tc_id: unique ID string (e.g., "TC-REQ-001-01")
- title: concise descriptive title
- requirement_id: matching requirement ID (e.g., "REQ-001")
- category: "Functional", "Negative", "Edge_Case", or "Security"
- preconditions: array of strings
- steps: array of action strings (e.g., ["Navigate to login page", "Enter valid credentials", "Click submit"])
- expected_results: array of expected outcome strings
- priority: "High", "Medium", or "Low"

JSON Array:"""

        def _mock_test_design() -> str:
            test_cases = []
            tc_num = 1
            for req in requirements:
                req_id = req.get("id", f"REQ-{tc_num:03d}")
                desc = req.get("description", "Feature specification")
                
                # Positive TC
                test_cases.append({
                    "tc_id": f"TC-{req_id}-01",
                    "title": f"Verify positive path for {desc[:40]}",
                    "requirement_id": req_id,
                    "category": "Functional",
                    "preconditions": [f"Target URL is accessible at {target_url}"],
                    "steps": [
                        f"Open browser and navigate to {target_url}",
                        "Interact with required input elements",
                        "Click submit/action button"
                    ],
                    "expected_results": [
                        "Page responds without HTTP errors",
                        "Success message or valid state is rendered"
                    ],
                    "priority": req.get("priority", "High")
                })
                tc_num += 1

                # Negative TC
                test_cases.append({
                    "tc_id": f"TC-{req_id}-02",
                    "title": f"Verify error handling for invalid input in {desc[:35]}",
                    "requirement_id": req_id,
                    "category": "Negative",
                    "preconditions": [f"Target URL is accessible at {target_url}"],
                    "steps": [
                        f"Open browser and navigate to {target_url}",
                        "Submit blank or invalid input values",
                        "Click submit"
                    ],
                    "expected_results": [
                        "Validation error indicator is displayed",
                        "Form submission is prevented"
                    ],
                    "priority": "Medium"
                })
                tc_num += 1

            return json.dumps(test_cases, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Senior Test Engineer specializing in end-to-end web test case design.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_mock_test_design
            )

            test_cases = self._parse_json_response(response_text)
            if not isinstance(test_cases, list) or len(test_cases) == 0:
                test_cases = json.loads(_mock_test_design())

            state["test_cases"] = test_cases
            if "metrics" in state and isinstance(state["metrics"], dict):
                state["metrics"]["total_test_cases"] = len(test_cases)

            self.log_success(f"Generated {len(test_cases)} structured test cases across {len(requirements)} requirements")
            for tc in test_cases:
                print(f" • [{tc.get('tc_id')}] ({tc.get('category')}): {tc.get('title')}")

            return state
        except Exception as e:
            self.log_error(f"Error during test case design: {e}")
            return state
