"""
Risk Coverage Planning Agent - Prioritizes test effort and identifies high-risk application areas
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class RiskCoveragePlanningAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Risk_Coverage_Planner", model_client)

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Planning test coverage and identifying high-risk modules...")
        requirements = state.get("validated_requirements", state.get("requirements", []))
        if not requirements:
            self.log_error("No requirements available for risk and coverage planning")
            return state

        req_summary = json.dumps(requirements, indent=2)
        prompt = f"""Analyze the validated requirements and generate a Risk and Test Coverage Plan.

Validated Requirements:
{req_summary}

Provide a JSON object containing:
1. "high_risk_areas": list of requirement IDs or features rated Critical or High risk
2. "testing_priorities": array of objects with fields:
   - requirement_id: string
   - priority_rank: integer (1 being highest priority)
   - coverage_types: list of strings (e.g. ["functional", "negative", "edge_case", "security"])
   - rationale: brief explanation of risk factor
3. "test_plan_summary": brief overall strategy summary string

Return ONLY valid JSON:"""

        def _mock_risk_plan() -> str:
            high_risk = [r["id"] for r in requirements if r.get("risk_level") in ["Critical", "High"]]
            if not high_risk and requirements:
                high_risk = [requirements[0]["id"]]

            priorities = []
            for rank, req in enumerate(requirements, start=1):
                p_types = ["functional"]
                if req.get("risk_level") in ["Critical", "High"]:
                    p_types.extend(["negative", "security", "edge_case"])
                else:
                    p_types.append("negative")

                priorities.append({
                    "requirement_id": req.get("id"),
                    "priority_rank": rank,
                    "coverage_types": p_types,
                    "rationale": f"Requirement rated {req.get('priority')} priority and {req.get('risk_level')} risk"
                })

            return json.dumps({
                "high_risk_areas": high_risk,
                "testing_priorities": priorities,
                "test_plan_summary": f"Risk-based testing strategy emphasizing {len(high_risk)} critical security and authentication modules."
            }, indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Quality Assurance Architect designing risk-based testing strategies.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_mock_risk_plan
            )

            plan_data = self._parse_json_response(response_text)
            if not isinstance(plan_data, dict):
                plan_data = json.loads(_mock_risk_plan())

            state["high_risk_areas"] = plan_data.get("high_risk_areas", [])
            state["testing_priorities"] = plan_data.get("testing_priorities", [])
            state["test_plan"] = plan_data

            self.log_success(f"Risk planning complete. Identified {len(state['high_risk_areas'])} high-risk areas.")
            return state
        except Exception as e:
            self.log_error(f"Error during risk and coverage planning: {e}")
            return state
