"""
Pytest unit tests for individual agents, state management, and PURE dataset loader
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pytest
import asyncio
from state import create_initial_state
from config import config
from agents import (
    RequirementExtractionAgent,
    RequirementValidationAgent,
    RiskCoveragePlanningAgent,
    TestDesignAgent as TDesignAgent,
    ReviewSecurityAgent,
    ScriptGenerationAgent,
    FailureAnalysisAgent
)

@pytest.fixture
def sample_state():
    srs_text = "The system shall allow registered users to log in securely using email and password."
    return create_initial_state(srs_text, "http://127.0.0.1:8080")

def test_pure_dataset_catalog():
    catalog = config.get_pure_catalog()
    assert len(catalog) > 0, "PURE dataset catalog should contain documents"
    first = catalog[0]
    assert "id" in first
    assert "title" in first
    assert "domain" in first
    assert first["requirements_count"] > 0

def test_pure_dataset_load():
    content = config.load_pure_document("2008_keepass")
    assert content is not None
    assert "KEEPASS" in content.upper()

def test_requirement_extraction_agent(sample_state):
    agent = RequirementExtractionAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "requirements" in updated_state
    assert len(updated_state["requirements"]) > 0
    assert updated_state["requirements"][0]["id"].startswith("REQ-")

def test_requirement_validation_agent(sample_state):
    sample_state["requirements"] = [
        {
            "id": "REQ-001",
            "description": "The system shall allow registered users to log in securely using valid credentials.",
            "priority": "High",
            "risk_level": "Critical",
            "testable": True,
            "category": "Security"
        },
        {
            "id": "REQ-002",
            "description": "The system shall be fast and user-friendly for all operations.",
            "priority": "Medium",
            "risk_level": "Medium",
            "testable": False,
            "category": "Performance"
        }
    ]
    agent = RequirementValidationAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "validated_requirements" in updated_state
    assert "quality_metrics" in updated_state
    metrics = updated_state["quality_metrics"]
    assert "coverage_gaps" in metrics
    assert "conflicts" in metrics
    assert "duplicate_pairs" in metrics
    assert "suggested_rewrites" in metrics
    assert len(metrics["suggested_rewrites"]) > 0
    # REQ-002 has ambiguous terms 'fast' and 'user-friendly'
    v_req2 = next(r for r in updated_state["validated_requirements"] if r["id"] == "REQ-002")
    assert v_req2["is_valid"] is False
    assert len(v_req2["validation_issues"]) > 0

def test_risk_coverage_planning_agent(sample_state):
    sample_state["validated_requirements"] = [
        {
            "id": "REQ-001",
            "description": "The system shall allow registered users to log in securely using password auth.",
            "category": "Security",
            "priority": "High",
            "risk_level": "Critical",
            "testable": True,
            "is_valid": True
        },
        {
            "id": "REQ-002",
            "description": "The system shall render user profile view with custom avatar.",
            "category": "UI",
            "priority": "Low",
            "risk_level": "Low",
            "testable": True,
            "is_valid": True
        },
        {
            "id": "REQ-003",
            "description": "The system shall update user account settings upon submission.",
            "category": "Functional",
            "priority": "Medium",
            "risk_level": "Medium",
            "testable": True,
            "is_valid": True
        }
    ]
    agent = RiskCoveragePlanningAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "risk_matrix" in updated_state
    matrix = updated_state["risk_matrix"]
    assert len(matrix) > 0

    for entry in matrix:
        assert "requirement_id" in entry, "risk_matrix entry must have requirement_id"
        assert entry["requirement_id"] != "", "requirement_id must not be empty"
        assert "risk_level" in entry, "risk_matrix entry must have risk_level"
        assert "tier" in entry, "risk_matrix entry must have tier"
        assert entry["tier"] in (1, 2, 3), f"tier must be 1, 2, or 3, got {entry['tier']}"

    # Verify REQ-001 is Tier 1, REQ-002 is Tier 3, REQ-003 is Tier 2
    req1_entry = next(e for e in matrix if e["requirement_id"] == "REQ-001")
    assert req1_entry["tier"] == 1
    req2_entry = next(e for e in matrix if e["requirement_id"] == "REQ-002")
    assert req2_entry["tier"] == 3
    req3_entry = next(e for e in matrix if e["requirement_id"] == "REQ-003")
    assert req3_entry["tier"] == 2

def test_test_design_agent(sample_state):
    sample_state["validated_requirements"] = [
        {
            "id": "REQ-001",
            "description": "The system shall allow registered users to log in securely.",
            "category": "Security",
            "priority": "High",
            "risk_level": "Critical",
            "testable": True
        },
        {
            "id": "REQ-002",
            "description": "The system shall update profile picture.",
            "category": "UI",
            "priority": "Low",
            "risk_level": "Low",
            "testable": True
        }
    ]
    sample_state["risk_matrix"] = [
        {"requirement_id": "REQ-001", "tier": 1, "severity": "Critical", "risk_level": "Critical", "area": "Auth", "likelihood": "Medium", "risk_score": 9, "mitigation": "TLS"},
        {"requirement_id": "REQ-002", "tier": 3, "severity": "Medium", "risk_level": "Medium", "area": "UI", "likelihood": "Low", "risk_score": 4, "mitigation": "DOM assertions"}
    ]
    sample_state["testing_priorities"] = [
        {"tier": "Tier 1", "target_requirements": ["REQ-001"], "focus": "Security"},
        {"tier": "Tier 3", "target_requirements": ["REQ-002"], "focus": "UI"}
    ]
    agent = TDesignAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "test_cases" in updated_state
    tcs = updated_state["test_cases"]
    assert len(tcs) > 0

    # Every test case must have a tier
    for tc in tcs:
        assert "tier" in tc, "Test case must have tier field"
        assert tc["tier"] in (1, 2, 3)

    # Test cases must be sorted by tier
    tiers = [tc["tier"] for tc in tcs]
    assert tiers == sorted(tiers), "Test cases must be sorted by tier"

    # Tier 1 requirement (REQ-001) gets 4 test types
    req1_tcs = [tc for tc in tcs if tc["requirement_id"] == "REQ-001"]
    req1_types = {tc["type"] for tc in req1_tcs}
    assert {"Positive", "Negative", "Security", "Boundary"}.issubset(req1_types)
    assert len(req1_tcs) == 4

    # Tier 3 requirement (REQ-002) gets 1 test type (Positive)
    req2_tcs = [tc for tc in tcs if tc["requirement_id"] == "REQ-002"]
    assert len(req2_tcs) == 1
    assert req2_tcs[0]["type"] == "Positive"

def test_script_generation_agent_tier_sorting(sample_state):
    sample_state["approved_test_cases"] = [
        {"id": "TC-003", "title": "Profile UI Test", "tier": 3, "type": "Positive", "requirement_id": "REQ-003"},
        {"id": "TC-001", "title": "Admin Login Security", "tier": 1, "type": "Security", "requirement_id": "REQ-001"},
        {"id": "TC-002", "title": "Order Processing", "tier": 2, "type": "Positive", "requirement_id": "REQ-002"},
    ]
    agent = ScriptGenerationAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "current_script" in updated_state
    script = updated_state["current_script"]
    assert "[Tier 1]" in script
    # Tier 1 test should precede Tier 3 test in TEST_CASES
    idx_tier1 = script.find("[Tier 1]")
    idx_tier3 = script.find("[Tier 3]")
    assert idx_tier1 < idx_tier3, "Tier 1 test scenario must execute before Tier 3 scenario in script"

def test_traceability_matrix_builder():
    from pipeline import build_traceability
    reqs = [
        {"id": "REQ-001", "description": "User login security"},
        {"id": "REQ-002", "description": "Data export reporting"},
        {"id": "REQ-003", "description": "Uncovered feature"},
    ]
    tcs = [
        {"id": "TC-001", "requirement_id": "REQ-001", "title": "Verify Login"},
        {"id": "TC-002", "requirement_id": "REQ-002", "title": "Verify Export"},
    ]
    results = [
        {"test_case_id": "TC-001", "status": "PASSED"},
        {"test_case_id": "TC-002", "status": "FAILED"},
    ]
    table = build_traceability(reqs, tcs, results)
    assert len(table) == 3
    t1 = next(t for t in table if t["requirement_id"] == "REQ-001")
    assert t1["status"] == "VERIFIED"
    t2 = next(t for t in table if t["requirement_id"] == "REQ-002")
    assert t2["status"] == "FAILED"
    t3 = next(t for t in table if t["requirement_id"] == "REQ-003")
    assert t3["status"] == "NOT_COVERED"

def test_failure_analysis_agent(sample_state):
    sample_state["last_error"] = "Timeout 5000ms exceeded waiting for locator('#submit')"
    agent = FailureAnalysisAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "error_analysis" in updated_state
    assert updated_state["error_analysis"]["root_cause"] is not None

def test_review_security_agent_coverage_rules(sample_state):
    sample_state["risk_matrix"] = [
        {"requirement_id": "REQ-001", "tier": 1, "severity": "Critical"},
        {"requirement_id": "REQ-002", "tier": 2, "severity": "High"},
        {"requirement_id": "REQ-003", "tier": 3, "severity": "Medium"},
    ]
    # REQ-001 (Tier 1): only Positive -> should be REJECTED
    # REQ-002 (Tier 2): has Negative -> should be APPROVED
    # REQ-003 (Tier 3): only Positive -> should be APPROVED
    sample_state["test_cases"] = [
        {"id": "TC-001", "requirement_id": "REQ-001", "type": "Positive", "tier": 1, "title": "Req1 Positive"},
        {"id": "TC-002", "requirement_id": "REQ-002", "type": "Positive", "tier": 2, "title": "Req2 Positive"},
        {"id": "TC-002B", "requirement_id": "REQ-002", "type": "Negative", "tier": 2, "title": "Req2 Negative"},
        {"id": "TC-003", "requirement_id": "REQ-003", "type": "Positive", "tier": 3, "title": "Req3 Positive"},
    ]
    agent = ReviewSecurityAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    approved_tcs = updated_state["approved_test_cases"]
    approved_ids = [tc["id"] for tc in approved_tcs]

    # TC-001 must NOT be approved because Tier 1 lacks Negative/Security
    assert "TC-001" not in approved_ids
    # TC-002, TC-002B, TC-003 must be approved
    assert "TC-002" in approved_ids
    assert "TC-002B" in approved_ids
    assert "TC-003" in approved_ids

    # Quality score should be (3 / 4) * 100 = 75.0
    audit = updated_state["security_audit"]
    assert audit["compliance_score"] == 75.0
    # There should be a security finding explaining the rejection
    assert any("REQ-001" in f for f in audit["owasp_findings"])

