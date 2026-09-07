"""
State schema for LangGraph and Agentic AI pipeline orchestration
"""
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

class AgentState(TypedDict, total=False):
    srs_input: str
    target_url: str
    
    # Requirement Extraction & Validation
    requirements: List[Dict[str, Any]]
    validated_requirements: List[Dict[str, Any]]
    validation_report: Optional[str]
    
    # Risk & Coverage Planning
    test_plan: Optional[Dict[str, Any]]
    high_risk_areas: Optional[List[str]]
    testing_priorities: Optional[List[Dict[str, Any]]]
    
    # Test Design & Review
    test_cases: List[Dict[str, Any]]
    review_results: Optional[Dict[str, Any]]
    approved_test_cases: Optional[List[Dict[str, Any]]]
    
    # Script Generation & Execution
    current_script: Optional[str]
    all_scripts: Optional[Dict[str, str]]
    execution_result: Optional[Dict[str, Any]]
    
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

def create_initial_state(srs_text: str, target_url: str = "") -> AgentState:
    from config import config
    url = target_url or config.TARGET_URL
    return {
        "srs_input": srs_text,
        "target_url": url,
        "requirements": [],
        "validated_requirements": [],
        "high_risk_areas": [],
        "testing_priorities": [],
        "test_cases": [],
        "approved_test_cases": [],
        "all_scripts": {},
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
