"""
Review & Security Agent - Audits test cases against OWASP security rules and approves suites for execution
"""
import json
from state import AgentState
from .base_agent import BaseAgent
from typing import List, Dict, Any

class ReviewSecurityAgent(BaseAgent):
    def __init__(self, model_client=None):
        super().__init__("Review_Security", model_client)

    def _review_test_cases(self, tcs: List[Dict[str, Any]], state: AgentState) -> Dict[str, Any]:
        """Review test cases against risk-tier coverage rules and OWASP standards.

        Rules:
        - Tier 1: Must have >= 1 Negative or Security test. If only Positive, reject all test cases for that requirement.
        - Tier 2: Must have >= 1 Negative test. If lacking, reject all test cases for that requirement.
        - Tier 3: Fine with just Positive.
        """
        # Read risk_matrix to get each requirement's tier
        tier_map: Dict[str, int] = {}
        for entry in state.get("risk_matrix", []):
            rid = entry.get("requirement_id")
            tier = entry.get("tier")
            if rid and tier is not None:
                try:
                    tier_map[rid] = int(tier)
                except (ValueError, TypeError):
                    pass

        # Fallback to testing_priorities
        for p in state.get("testing_priorities", []):
            tier_str = p.get("tier", "")
            t_val = 1 if "Tier 1" in tier_str else (2 if "Tier 2" in tier_str else 3)
            for rid in p.get("target_requirements", []):
                if rid not in tier_map:
                    tier_map[rid] = t_val

        # Group test cases by requirement_id
        req_tcs: Dict[str, List[Dict[str, Any]]] = {}
        for tc in tcs:
            rid = tc.get("requirement_id", "UNKNOWN")
            req_tcs.setdefault(rid, []).append(tc)

        approved_tc_ids: List[str] = []
        security_findings: List[str] = []

        for rid, req_test_cases in req_tcs.items():
            # Get requirement tier, defaulting to test case tier or Tier 2
            tier = tier_map.get(rid, req_test_cases[0].get("tier", 2))
            types_present = set(tc.get("type", "Positive") for tc in req_test_cases)

            if tier == 1:
                # Tier 1 must have at least one Negative or Security type test
                has_neg_or_sec = bool(types_present & {"Negative", "Security"})
                if not has_neg_or_sec:
                    security_findings.append(
                        f"Tier 1 Security Violation: Requirement '{rid}' lacks Negative or Security test coverage (found only {', '.join(types_present)}). Rejected all {len(req_test_cases)} test cases for this requirement."
                    )
                else:
                    approved_tc_ids.extend(tc.get("id") for tc in req_test_cases)

            elif tier == 2:
                # Tier 2 must have at least one Negative test
                has_neg = "Negative" in types_present
                if not has_neg:
                    security_findings.append(
                        f"Tier 2 Quality Violation: Requirement '{rid}' lacks Negative test coverage (found only {', '.join(types_present)}). Rejected all {len(req_test_cases)} test cases for this requirement."
                    )
                else:
                    approved_tc_ids.extend(tc.get("id") for tc in req_test_cases)

            else:
                # Tier 3 is fine with just Positive
                approved_tc_ids.extend(tc.get("id") for tc in req_test_cases)

        # Standard OWASP checks for additional security findings
        has_auth = any("auth" in tc.get("title", "").lower() or "login" in tc.get("title", "").lower() for tc in tcs)
        has_negative_any = any(tc.get("type") == "Negative" for tc in tcs)
        if has_auth and not has_negative_any:
            security_findings.append("OWASP A07:2021 (Identification Failures): Need negative credential stuffing & brute force tests.")

        total_count = len(tcs)
        approved_count = len(approved_tc_ids)
        quality_score = round((approved_count / total_count) * 100, 1) if total_count > 0 else 0.0

        recommendations = [
            "Sanitize all automated test inputs against SQL injection and XSS patterns.",
            "Verify session cookie HttpOnly, Secure, and SameSite attributes in Playwright contexts.",
            "Include timeout guardrails to prevent hanging on denial-of-service edge cases."
        ]

        return {
            "approved_ids": approved_tc_ids,
            "security_findings": security_findings,
            "compliance_score": quality_score,
            "recommendations": recommendations
        }

    async def invoke(self, state: AgentState) -> AgentState:
        self.log_start("Conducting OWASP security review and test case compliance audit...")
        tcs = state.get("test_cases", [])
        if not tcs:
            self.log_error("No test cases available to review")
            return state

        prompt = f"""Review the following test cases for quality completeness, safety, and security compliance (OWASP Top 10):
{json.dumps(tcs[:15], indent=2)}

Output a JSON object with:
- "approved_ids": list of test case IDs approved for automation execution
- "security_findings": array of strings describing any detected vulnerabilities or test gaps
- "compliance_score": integer between 0 and 100
- "recommendations": array of suggestions to harden tests and security coverage

Return ONLY valid JSON object:"""

        def _deterministic_security_review() -> str:
            return json.dumps(self._review_test_cases(tcs, state), indent=2)

        try:
            response_text = await self.call_llm(
                prompt=prompt,
                system_instruction="You are a Cybersecurity and QA Security Compliance Auditor.",
                model_name=self.config.GEMINI_FLASH_MODEL,
                mock_fallback=_deterministic_security_review
            )

            rev_data = self._parse_json_response(response_text)
            if not isinstance(rev_data, dict):
                rev_data = json.loads(_deterministic_security_review())

            # Enforce coverage rules against approvals
            rule_review = self._review_test_cases(tcs, state)
            rule_approved_ids = set(rule_review["approved_ids"])

            if "approved_ids" in rev_data and isinstance(rev_data["approved_ids"], list):
                approved_ids = set(rev_data["approved_ids"]) & rule_approved_ids
            else:
                approved_ids = rule_approved_ids

            approved_tcs = [tc for tc in tcs if tc.get("id") in approved_ids]
            compliance_score = round((len(approved_tcs) / len(tcs)) * 100, 1) if tcs else 0.0

            all_findings = rule_review["security_findings"] + [
                f for f in rev_data.get("security_findings", [])
                if f not in rule_review["security_findings"]
            ]

            state["approved_test_cases"] = approved_tcs
            state["security_audit"] = {
                "owasp_findings": all_findings,
                "compliance_score": compliance_score,
                "recommendations": rev_data.get("recommendations", rule_review["recommendations"])
            }
            state["review_results"] = {
                "total_reviewed": len(tcs),
                "total_approved": len(approved_tcs),
                "compliance_score": compliance_score
            }

            self.log_success(f"Audit complete: Approved {len(approved_tcs)}/{len(tcs)} test cases ({compliance_score}% quality score)")
            return state

        except Exception as e:
            self.log_error(f"Error during security review: {e}")
            fallback = self._review_test_cases(tcs, state)
            approved_ids = set(fallback["approved_ids"])
            approved_tcs = [tc for tc in tcs if tc.get("id") in approved_ids]
            state["approved_test_cases"] = approved_tcs
            state["security_audit"] = {
                "owasp_findings": fallback["security_findings"],
                "compliance_score": fallback["compliance_score"],
                "recommendations": fallback["recommendations"]
            }
            state["review_results"] = {
                "total_reviewed": len(tcs),
                "total_approved": len(approved_tcs),
                "compliance_score": fallback["compliance_score"]
            }
            return state
