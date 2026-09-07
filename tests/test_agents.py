"""
Pytest unit tests for individual agents and state management
"""
import pytest
import asyncio
from state import create_initial_state
from agents import (
    RequirementExtractionAgent,
    RequirementValidationAgent,
    RiskCoveragePlanningAgent,
    TestDesignAgent,
    ReviewSecurityAgent,
    ScriptGenerationAgent,
    FailureAnalysisAgent
)

@pytest.fixture
def sample_state():
    srs_text = "The user shall log in using email and password."
    return create_initial_state(srs_text, "https://demo.example.com")

@pytest.mark.asyncio
async def test_requirement_extraction_agent(sample_state):
    agent = RequirementExtractionAgent()
    updated_state = await agent.invoke(sample_state)
    assert "requirements" in updated_state
    assert len(updated_state["requirements"]) > 0
    assert updated_state["requirements"][0]["id"].startswith("REQ-")

@pytest.mark.asyncio
async def test_requirement_validation_agent(sample_state):
    sample_state["requirements"] = [{
        "id": "REQ-001",
        "description": "User login specification",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True
    }]
    agent = RequirementValidationAgent()
    updated_state = await agent.invoke(sample_state)
    assert "validated_requirements" in updated_state
    assert updated_state["validated_requirements"][0]["is_valid"] is True

@pytest.mark.asyncio
async def test_risk_coverage_planning_agent(sample_state):
    sample_state["validated_requirements"] = [{
        "id": "REQ-001",
        "description": "User login specification",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True,
        "is_valid": True
    }]
    agent = RiskCoveragePlanningAgent()
    updated_state = await agent.invoke(sample_state)
    assert "high_risk_areas" in updated_state
    assert "testing_priorities" in updated_state

@pytest.mark.asyncio
async def test_test_design_agent(sample_state):
    sample_state["validated_requirements"] = [{
        "id": "REQ-001",
        "description": "User login specification",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True
    }]
    agent = TestDesignAgent()
    updated_state = await agent.invoke(sample_state)
    assert "test_cases" in updated_state
    assert len(updated_state["test_cases"]) > 0

@pytest.mark.asyncio
async def test_failure_analysis_agent(sample_state):
    sample_state["last_error"] = "Timeout 5000ms exceeded waiting for locator('#submit')"
    agent = FailureAnalysisAgent()
    updated_state = await agent.invoke(sample_state)
    assert "error_analysis" in updated_state
    assert updated_state["error_analysis"]["root_cause"] is not None
