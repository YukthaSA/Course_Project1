"""
Risk & Coverage Planning Agent - Analyzes risks, generates 2D risk matrix, and prioritizes test suites
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class RiskCoveragePlanningAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Risk_Coverage_Planning", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Analyzing requirement risk levels and designing test coverage priorities...")
        validated_reqs = state.get("validated_requirements", state.get("requirements", []))
        if not validated_reqs:
            self.log_error("No validated requirements available for risk analysis")
            return state

        reqs_summary = json.dumps([
            {"id": r.get("id"), "desc": r.get("description"), "category": r.get("category"), "priority": r.get("priority")}
            for r in validated_reqs[:20]
        ], indent=2)

        prompt = f"""Conduct a formal Risk Assessment and Test Coverage Plan for the following requirements:
{reqs_summary}

Generate a JSON object with:
1. "high_risk_areas": list of critical functional subsystems or vulnerability areas (strings)
2. "risk_matrix": array of objects, each containing:
   - "area": module name or requirement focus
   - "severity": "Critical", "High", "Medium", or "Low"
   - "likelihood": "High", "Medium", or "Low"
   - "risk_score": integer (1-10)
   - "mitigation": recommended automated testing strategy
3. "testing_priorities": array of prioritized test suites with:
   - "tier": "Tier 1 (Smoke / Blocker)", "Tier 2 (Core Business Logic)", or "Tier 3 (Edge Cases & Security)"
   - "target_requirements": list of requirement IDs
   - "focus": testing focus description

Return ONLY valid JSON object:"""

        def _deterministic_risk_planner() -> str:
            high_risk_areas = []
            risk_matrix = []
            priorities = []

            tier1_reqs = []
            tier2_reqs = []
            tier3_reqs = []

            for r in validated_reqs:
                rid = r.get("id", "REQ")
                cat = r.get("category", "Functional")
                desc = r.get("description", "")
                desc_lower = desc.lower()

                if cat in ["Security", "Safety"] or any(w in desc_lower for w in ["auth", "login", "password", "encrypt", "emergency", "brake"]):
                    if "Authentication & Security Integrity" not in high_risk_areas:
                        high_risk_areas.append("Authentication & Security Integrity")
                    tier1_reqs.append(rid)
                    risk_matrix.append({
                        "area": f"{rid} - {cat} Module",
                        "severity": "Critical",
                        "likelihood": "Medium",
                        "risk_score": 9,
                        "mitigation": "Automated negative input testing, boundary validation, and TLS verification."
                    })
                elif cat in ["Performance", "UI"]:
                    tier3_reqs.append(rid)
                    risk_matrix.append({
                        "area": f"{rid} - {cat} Experience",
                        "severity": "Medium",
                        "likelihood": "High",
                        "risk_score": 6,
                        "mitigation": "Playwright DOM selector assertions and async response timing thresholds."
                    })
                else:
                    tier2_reqs.append(rid)
                    risk_matrix.append({
                        "area": f"{rid} - Core Workflow",
                        "severity": "High",
                        "likelihood": "Medium",
                        "risk_score": 7,
                        "mitigation": "End-to-end positive user workflow validation and state transitions."
                    })

            if not high_risk_areas:
                high_risk_areas = ["Input Validation & Data Integrity", "State Synchronization", "Error Handling & Recovery"]

            priorities = [
                {
                    "tier": "Tier 1 (Smoke / Blocker)",
                    "target_requirements": tier1_reqs if tier1_reqs else [r.get("id") for r in validated_reqs[:2]],
                    "focus": "Security barriers, access control, and fatal crash prevention"
                },
                {
                    "tier": "Tier 2 (Core Business Logic)",
                    "target_requirements": tier2_reqs if tier2_reqs else [r.get("id") for r in validated_reqs[2:6]],
                    "focus": "Primary functional flow and end-to-end user actions"
                },
                {
                    "tier": "Tier 3 (Edge Cases & UI)",
                    "target_requirements": tier3_reqs if tier3_reqs else [r.get("id") for r in validated_reqs[6:]],
                    "focus": "Boundary values, error popups, and UI responsiveness"
                }
            ]

            return json.dumps({
                "high_risk_areas": high_risk_areas,
                "risk_matrix": risk_matrix[:10],
                "testing_priorities": priorities
            }, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Principal QA Architect designing risk-based test coverage plans.",
                model_name=self.config.GEMINI_PRO_MODEL,
                mock_fallback=_deterministic_risk_planner
            )

            plan_data = self._parse_json_response(response_text)
            if not isinstance(plan_data, dict):
                plan_data = json.loads(_deterministic_risk_planner())

            state["high_risk_areas"] = plan_data.get("high_risk_areas", ["Security Integrity", "Core Business Flow"])
            state["risk_matrix"] = plan_data.get("risk_matrix", [])
            state["testing_priorities"] = plan_data.get("testing_priorities", [])
            state["test_plan"] = {
                "total_high_risk_items": len(state["high_risk_areas"]),
                "coverage_goal": "100% of Tier 1 & Tier 2 Requirements",
                "recommended_execution_mode": "Automated End-to-End Playwright Suite"
            }

            self.log_success(f"Risk analysis complete: {len(state['high_risk_areas'])} high-risk areas identified.")
            return state
        except Exception as e:
            self.log_error(f"Error during risk planning: {e}")
            fallback = json.loads(_deterministic_risk_planner())
            state["high_risk_areas"] = fallback["high_risk_areas"]
            state["risk_matrix"] = fallback["risk_matrix"]
            state["testing_priorities"] = fallback["testing_priorities"]
            return state
