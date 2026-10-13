"""
Test Design Agent - Converts validated requirements into structured test cases and BDD / Gherkin scenarios
Tier-aware: reads tiers from the risk agent output and generates weighted test cases sorted by tier.
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

# Tier 1 → 4 test types, Tier 2 → 2, Tier 3 → 1
TIER_TEST_TYPES: Dict[int, List[str]] = {
    1: ["Positive", "Negative", "Security", "Boundary"],
    2: ["Positive", "Negative"],
    3: ["Positive"],
}


class TestDesignAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Test_Design", model_client)

    # ------------------------------------------------------------------
    # Build a {requirement_id: tier} lookup from the risk agent's output
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tier_lookup(state: AgentState) -> Dict[str, int]:
        """Return {req_id: tier_int} from the risk_matrix produced by RiskCoveragePlanningAgent."""
        lookup: Dict[str, int] = {}

        # Primary source: risk_matrix entries with explicit requirement_id + tier
        for entry in state.get("risk_matrix", []):
            rid = entry.get("requirement_id", "")
            tier = entry.get("tier")
            if rid and tier is not None:
                lookup[rid] = int(tier)

        # Fallback: testing_priorities (tier text → tier int mapping)
        tier_label_map = {
            "Tier 1": 1, "Tier 2": 2, "Tier 3": 3,
            "Tier 1 (Smoke / Blocker)": 1,
            "Tier 2 (Core Business Logic)": 2,
            "Tier 3 (Edge Cases & Security)": 3,
            "Tier 3 (Edge Cases & UI)": 3,
        }
        for p in state.get("testing_priorities", []):
            tier_str = p.get("tier", "")
            tier_int = None
            for label, val in tier_label_map.items():
                if label in tier_str:
                    tier_int = val
                    break
            if tier_int is None:
                tier_int = 2  # default
            for rid in p.get("target_requirements", []):
                if rid not in lookup:
                    lookup[rid] = tier_int

        return lookup

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Designing tier-weighted test cases and BDD / Gherkin specifications...")
        reqs = state.get("validated_requirements", state.get("requirements", []))
        if not reqs:
            self.log_error("No requirements found to design test cases")
            return state

        tier_lookup = self._build_tier_lookup(state)

        prompt = f"""Design comprehensive test cases based on these requirements:
{json.dumps(reqs[:15], indent=2)}

For each requirement, generate structured test cases including positive paths, negative paths, and security/boundary tests.
Provide a JSON array where each object has:
- id: "TC-001", "TC-002", etc.
- title: clear summary of test scenario
- type: "Positive", "Negative", "Security", or "Boundary"
- requirement_id: corresponding requirement ID
- tier: integer (1, 2, or 3) indicating risk priority — 1 is highest
- preconditions: string or array of setup conditions
- steps: array of action steps to perform
- expected_result: expected system outcome
- gherkin: BDD Gherkin scenario string (Scenario, Given, When, Then)

Return ONLY valid JSON array:"""

        def _deterministic_test_designer() -> str:
            test_cases: List[Dict[str, Any]] = []
            tc_num = 1

            # Sort reqs by tier (Tier 1 first) so IDs are sequential within tiers
            def _req_tier(r: dict) -> int:
                return tier_lookup.get(r.get("id", ""), 2)

            sorted_reqs = sorted(reqs[:15], key=_req_tier)

            for r in sorted_reqs:
                rid = r.get("id", f"REQ-{tc_num:03d}")
                desc = r.get("description", "System feature verification")
                cat = r.get("category", "Functional")
                tier = tier_lookup.get(rid, 2)
                test_types = TIER_TEST_TYPES.get(tier, ["Positive"])

                for ttype in test_types:
                    tc: Dict[str, Any] = {
                        "id": f"TC-{tc_num:03d}",
                        "title": self._make_title(ttype, desc, rid),
                        "type": ttype,
                        "requirement_id": rid,
                        "tier": tier,
                        "preconditions": "Application is accessible and in ready state",
                        "steps": self._make_steps(ttype, rid, desc),
                        "expected_result": self._make_expected(ttype, desc),
                        "gherkin": self._make_gherkin(ttype, rid, desc),
                    }
                    test_cases.append(tc)
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

            # --- Backfill tier on LLM-generated test cases if missing ---
            for tc in tc_list:
                if "tier" not in tc or tc["tier"] is None:
                    tc["tier"] = tier_lookup.get(tc.get("requirement_id", ""), 2)

            # --- Sort by tier (Tier 1 first) ---
            tc_list.sort(key=lambda tc: tc.get("tier", 2))

            state["test_cases"] = tc_list
            state["gherkin_scenarios"] = [tc.get("gherkin", "") for tc in tc_list if tc.get("gherkin")]
            if "metrics" in state and isinstance(state["metrics"], dict):
                state["metrics"]["total_test_cases"] = len(tc_list)

            tier_counts = {}
            for tc in tc_list:
                t = tc.get("tier", 2)
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tier_summary = ", ".join(f"Tier {t}: {c}" for t, c in sorted(tier_counts.items()))
            self.log_success(f"Generated {len(tc_list)} test cases sorted by tier ({tier_summary})")
            return state
        except Exception as e:
            self.log_error(f"Error designing test cases: {e}")
            fallback = json.loads(_deterministic_test_designer())
            state["test_cases"] = fallback
            state["gherkin_scenarios"] = [tc.get("gherkin", "") for tc in fallback if tc.get("gherkin")]
            return state

    # ------------------------------------------------------------------
    # Helper methods for deterministic test-case generation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_title(ttype: str, desc: str, rid: str) -> str:
        prefixes = {
            "Positive": f"Verify Successful {desc[:50]}",
            "Negative": f"Validate Negative Input Handling for {rid}",
            "Security": f"Security Boundary Test for {rid}",
            "Boundary": f"Edge/Boundary Value Test for {rid}",
        }
        return prefixes.get(ttype, f"Test {rid}")

    @staticmethod
    def _make_steps(ttype: str, rid: str, desc: str) -> List[str]:
        if ttype == "Positive":
            return [
                "Navigate to application interface",
                f"Submit valid inputs adhering to {rid}",
                "Observe system confirmation and response",
            ]
        elif ttype == "Negative":
            return [
                "Navigate to target input endpoint",
                "Submit invalid credentials or malformed parameters",
                "Verify rejection message and absence of unauthorized access",
            ]
        elif ttype == "Security":
            return [
                "Attempt injection / XSS payloads on input fields",
                f"Verify {rid} rejects dangerous input safely",
                "Check response headers for security best practices",
            ]
        else:  # Boundary
            return [
                f"Submit minimum and maximum boundary values for {rid}",
                "Submit values just outside valid range",
                "Verify correct acceptance/rejection at boundaries",
            ]

    @staticmethod
    def _make_expected(ttype: str, desc: str) -> str:
        if ttype == "Positive":
            return f"System completes action according to specification: {desc[:60]}"
        elif ttype == "Negative":
            return "System safely rejects invalid input with descriptive error message without crashing"
        elif ttype == "Security":
            return "System blocks malicious input and returns appropriate error without data leakage"
        else:
            return "System correctly handles boundary values — accepts valid, rejects out-of-range"

    @staticmethod
    def _make_gherkin(ttype: str, rid: str, desc: str) -> str:
        if ttype == "Positive":
            return (f"Scenario: Successfully verify {rid}\n"
                    f"  Given the user is on the application home page\n"
                    f"  When valid parameters matching '{desc[:40]}' are entered\n"
                    f"  Then the system responds with success status")
        elif ttype == "Negative":
            return (f"Scenario: Negative validation for {rid}\n"
                    f"  Given the application is accessible\n"
                    f"  When malformed or unauthorized inputs are supplied\n"
                    f"  Then an appropriate error indication is displayed and access is blocked")
        elif ttype == "Security":
            return (f"Scenario: Security test for {rid}\n"
                    f"  Given the application is accessible\n"
                    f"  When injection or XSS payloads are submitted\n"
                    f"  Then the system sanitizes input and returns no sensitive data")
        else:
            return (f"Scenario: Boundary test for {rid}\n"
                    f"  Given the application is accessible\n"
                    f"  When extreme boundary values are submitted\n"
                    f"  Then the system accepts valid boundaries and rejects invalid ones")
