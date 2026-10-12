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
    sample_state["requirements"] = [{
        "id": "REQ-001",
        "description": "The system shall allow registered users to log in securely using valid credentials.",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True
    }]
    agent = RequirementValidationAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "validated_requirements" in updated_state
    assert updated_state["validated_requirements"][0]["is_valid"] is True

def test_risk_coverage_planning_agent(sample_state):
    sample_state["validated_requirements"] = [{
        "id": "REQ-001",
        "description": "The system shall allow registered users to log in securely using valid credentials.",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True,
        "is_valid": True
    }]
    agent = RiskCoveragePlanningAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "high_risk_areas" in updated_state
    assert "testing_priorities" in updated_state

def test_test_design_agent(sample_state):
    sample_state["validated_requirements"] = [{
        "id": "REQ-001",
        "description": "The system shall allow registered users to log in securely using valid credentials.",
        "priority": "High",
        "risk_level": "Critical",
        "testable": True
    }]
    agent = TDesignAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "test_cases" in updated_state
    assert len(updated_state["test_cases"]) > 0

def test_failure_analysis_agent(sample_state):
    sample_state["last_error"] = "Timeout 5000ms exceeded waiting for locator('#submit')"
    agent = FailureAnalysisAgent()
    updated_state = asyncio.run(agent.invoke(sample_state))
    assert "error_analysis" in updated_state
    assert updated_state["error_analysis"]["root_cause"] is not None
