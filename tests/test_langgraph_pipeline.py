"""
Unit tests for LangGraph StateGraph pipeline compilation and execution topology.
"""
import pytest
from langgraph.graph import StateGraph
from pipeline import AgenticQAPipeline
from state import AgentState


def test_langgraph_workflow_compilation():
    """Verify that AgenticQAPipeline builds and compiles a valid LangGraph StateGraph."""
    pipeline = AgenticQAPipeline()
    compiled_graph = pipeline.build_graph()
    assert compiled_graph is not None

    # Verify nodes exist in the LangGraph graph
    graph_repr = compiled_graph.get_graph()
    node_names = set(graph_repr.nodes.keys())
    expected_nodes = {
        "site_probe",
        "extract_requirements",
        "validate_requirements",
        "plan_risk_coverage",
        "design_tests",
        "review_security",
        "generate_script",
        "execute_playwright",
        "analyze_failure",
        "finalize"
    }
    assert expected_nodes.issubset(node_names)


def test_langgraph_mermaid_diagram_generation():
    """Verify that the LangGraph workflow can export a Mermaid diagram."""
    pipeline = AgenticQAPipeline()
    mermaid_str = pipeline.get_graph_mermaid()
    assert isinstance(mermaid_str, str)
    assert len(mermaid_str) > 50
    assert "graph TD" in mermaid_str or "flowchart" in mermaid_str
    assert "site_probe" in mermaid_str
    assert "analyze_failure" in mermaid_str
