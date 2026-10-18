"""
State schema for Agentic AI pipeline orchestration
Enhanced for real-world PURE dataset documents, BDD test cases, and multi-stage QA metrics
"""
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

class AgentState(TypedDict, total=False):
    srs_input: str
    target_url: str
    document_metadata: Optional[Dict[str, Any]]
    
    # Requirement Extraction & Validation
    requirements: List[Dict[str, Any]]
    validated_requirements: List[Dict[str, Any]]
    validation_report: Optional[str]
    quality_metrics: Optional[Dict[str, Any]]  # testability_score, ambiguity_count, etc.
    
    # Risk & Coverage Planning
    test_plan: Optional[Dict[str, Any]]
    high_risk_areas: Optional[List[str]]
    testing_priorities: Optional[List[Dict[str, Any]]]
    risk_matrix: Optional[List[Dict[str, Any]]]
    
    # Test Design & Review
    test_cases: List[Dict[str, Any]]
    approved_test_cases: Optional[List[Dict[str, Any]]]
    gherkin_scenarios: Optional[List[str]]
    review_results: Optional[Dict[str, Any]]
    security_audit: Optional[Dict[str, Any]]
    
    # Script Generation & Execution
    current_script: Optional[str]
    all_scripts: Optional[Dict[str, str]]
    execution_result: Optional[Dict[str, Any]]

    # Per-Test-Case Results & Requirement Traceability
    test_results: Optional[List[Dict[str, Any]]]       # individual test outcomes
    traceability: Optional[List[Dict[str, Any]]]        # requirement → test → status mapping
    
    # Failure Analysis & Retry Control
    error_analysis: Optional[Dict[str, Any]]
    retry_count: int
    last_error: Optional[str]
    retry_history: Optional[List[Dict[str, Any]]]
    
    # Human-in-the-Loop
    human_review_required: bool
    human_review_context: Optional[Dict[str, Any]]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    
    # System Memory & Analytics
    learning_memory: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    final_results: Optional[Dict[str, Any]]

def create_initial_state(srs_text: str, target_url: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
    from config import config
    url = target_url or config.TARGET_URL
    return {
        "srs_input": srs_text,
        "target_url": url,
        "document_metadata": metadata or {
            "source": "PURE Benchmark Dataset",
            "ingested_at": datetime.now().isoformat()
        },
        "requirements": [],
        "validated_requirements": [],
        "quality_metrics": {
            "testability_score": 0,
            "ambiguous_terms_found": [],
            "clarity_rating": "Pending"
        },
        "high_risk_areas": [],
        "testing_priorities": [],
        "risk_matrix": [],
        "test_cases": [],
        "approved_test_cases": [],
        "gherkin_scenarios": [],
        "review_results": {},
        "security_audit": {
            "owasp_findings": [],
            "compliance_score": 100
        },
        "all_scripts": {},
        "test_results": [],
        "traceability": [],
        "retry_count": 0,
        "human_review_required": False,
        "retry_history": [],
        "learning_memory": {
            "common_failures": [],
            "successful_locators": {},
            "execution_rules": []
        },
        "metrics": {
            "start_time": datetime.now().isoformat(),
            "total_requirements": 0,
            "total_test_cases": 0,
            "passed_scripts": 0,
            "failed_scripts": 0,
            "human_interventions": 0,
        }
    }