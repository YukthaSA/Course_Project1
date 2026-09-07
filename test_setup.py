"""
Simple test to verify environment setup and imports
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("Testing Agentic QA Framework setup & dependencies...")

try:
    from config import config
    print("[OK] Config loaded")
except Exception as e:
    print(f"[FAIL] Config error: {e}")
    sys.exit(1)

try:
    from state import AgentState, create_initial_state
    print("[OK] State schema loaded")
except Exception as e:
    print(f"[FAIL] State error: {e}")
    sys.exit(1)

try:
    from agents import (
        RequirementExtractionAgent,
        RequirementValidationAgent,
        RiskCoveragePlanningAgent,
        TestDesignAgent,
        ReviewSecurityAgent,
        ScriptGenerationAgent,
        FailureAnalysisAgent
    )
    print("[OK] All 7 Agents loaded successfully")
except Exception as e:
    print(f"[FAIL] Agents error: {e}")
    sys.exit(1)

try:
    from engine import PlaywrightRunner
    print("[OK] Playwright Runner engine loaded")
except Exception as e:
    print(f"[FAIL] Engine error: {e}")
    sys.exit(1)

try:
    import google.generativeai as genai
    print("[OK] Google AI package available")
except Exception as e:
    print(f"[NOTICE] Google AI notice: {e}")

try:
    import streamlit as st
    print("[OK] Streamlit loaded")
except Exception as e:
    print(f"[FAIL] Streamlit error: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("[PASS] ALL SETUP CHECKS PASSED!")
print("=" * 50)
print()
print("Next steps:")
print("1. Update .env with your API keys (optional)")
print("2. Run pipeline integration test: python test_pipeline.py")
print("3. Launch dashboard: streamlit run app.py")
