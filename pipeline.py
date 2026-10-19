"""
Agentic QA Pipeline Orchestration - Connects specialized agents, execution engine, retry loops, and human-in-the-loop escalation
Enhanced with PURE benchmark dataset support and detailed quality analytics
"""
import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from config import config
from state import AgentState, create_initial_state
from agents import (
    RequirementExtractionAgent,
    RequirementValidationAgent,
    RiskCoveragePlanningAgent,
    TestDesignAgent,
    ReviewSecurityAgent,
    ScriptGenerationAgent,
    FailureAnalysisAgent
)
from engine import PlaywrightRunner

logger = logging.getLogger("PipelineOrchestrator")


def build_traceability(requirements, test_cases, test_results):
    """Build a requirement → test case → execution status traceability table.

    Returns a list of dicts:
      [{requirement_id, requirement_desc, test_case_id, test_title, status}]
    where status is VERIFIED / FAILED / NOT_COVERED / PARTIAL
    """
    # Index test results by test_case_id
    result_by_tc = {}
    for r in (test_results or []):
        tc_id = r.get("test_case_id", "")
        result_by_tc[tc_id] = r.get("status", "NOT_RUN")

    # Index test cases by requirement_id
    tc_by_req = {}
    for tc in (test_cases or []):
        req_id = tc.get("requirement_id", "")
        tc_by_req.setdefault(req_id, []).append(tc)

    table = []
    for req in (requirements or []):
        req_id = req.get("id", "")
        req_desc = req.get("description", "")[:80]
        linked_tcs = tc_by_req.get(req_id, [])

        if not linked_tcs:
            table.append({
                "requirement_id": req_id,
                "requirement_desc": req_desc,
                "test_case_id": "—",
                "test_title": "—",
                "status": "NOT_COVERED"
            })
        else:
            for tc in linked_tcs:
                tc_id = tc.get("id", "")
                tc_title = tc.get("title", "")
                exec_status = result_by_tc.get(tc_id, "NOT_RUN")

                if exec_status == "PASSED":
                    status = "VERIFIED"
                elif exec_status in ("FAILED", "ERROR"):
                    status = "FAILED"
                else:
                    status = "PARTIAL"

                table.append({
                    "requirement_id": req_id,
                    "requirement_desc": req_desc,
                    "test_case_id": tc_id,
                    "test_title": tc_title,
                    "status": status
                })

    return table

class AgenticQAPipeline:
    def __init__(self, model_client=None):
        self.req_extractor = RequirementExtractionAgent(model_client)
        self.req_validator = RequirementValidationAgent(model_client)
        self.risk_planner = RiskCoveragePlanningAgent(model_client)
        self.test_designer = TestDesignAgent(model_client)
        self.reviewer = ReviewSecurityAgent(model_client)
        self.script_generator = ScriptGenerationAgent(model_client)
        self.failure_analyzer = FailureAnalysisAgent(model_client)
        self.runner = PlaywrightRunner(timeout=config.TIMEOUT)

    async def run(
        self,
        srs_text: str,
        target_url: str = "",
        max_retries: int = config.MAX_RETRIES,
        status_callback=None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Run the complete Agentic QA Pipeline from SRS to Playwright Execution & Retry Loop.

        target_url: the real website / app to test. Empty or "demo" starts the built-in demo app
        (engine/demo_app.py) for the duration of the run - it is never used as a silent fallback.
        """
        target_url = (target_url or config.TARGET_URL or "").strip()
        demo_server = None
        if target_url.lower() in ("", "demo", "builtin"):
            from engine.demo_app import start_demo_server
            demo_server, port = start_demo_server(0)
            target_url = f"http://127.0.0.1:{port}"
        elif "://" not in target_url:
            target_url = "https://" + target_url
        run_dir = os.path.join(config.LOGS_DIR, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        try:
            return await self._run(srs_text, target_url, max_retries, status_callback, metadata, run_dir,
                                   "demo" if demo_server else "external")
        finally:
            if demo_server:
                demo_server.shutdown()

    async def _run(self, srs_text, target_url, max_retries, status_callback, metadata, run_dir, target_mode) -> AgentState:
        logger.info("============================================================")
        logger.info("🚀 STARTING AGENTIC QA PIPELINE EXECUTION")
        if metadata:
            logger.info(f"📄 Document: {metadata.get('title', 'Unknown')} ({metadata.get('domain', 'General')})")
        logger.info(f"🎯 Target: {target_url} ({target_mode})")
        logger.info("============================================================")

        state = create_initial_state(srs_text, target_url, metadata)
        state["run_dir"] = run_dir
        state["target_mode"] = target_mode

        def _update(msg: str, step: int):
            logger.info(f"[{step}/7] {msg}")
            if status_callback:
                status_callback(msg, step, state)

        # ----------------------------------------------------
        # Stage 0: Open the real target - is it up, is it healthy, what is on the page?
        # ----------------------------------------------------
        _update("Stage 0/7: Opening the target in a real browser and checking its health...", 0)
        health = await asyncio.to_thread(self.runner.probe_target, target_url, os.path.join(run_dir, "target"), config.HEADLESS)
        state["target_health"] = health
        state["target_profile"] = health.get("inventory") or {}
        if not health.get("reachable"):
            reason = health.get("error") or "no response"
            logger.error(f"❌ Target {target_url} is not reachable: {reason}")
            state["execution_result"] = {"status": "TARGET_UNREACHABLE", "error": f"{target_url} is not reachable: {reason}",
                                         "stdout": "", "stderr": "", "duration_seconds": 0, "screenshots": []}
            state["human_review_required"] = True
            state["human_review_context"] = {"reason": "target_unreachable", "last_error": reason, "total_attempts": 0}
            state["final_results"] = {
                "completed_at": datetime.now().isoformat(), "status": "TARGET_UNREACHABLE", "passed": False,
                "document_title": metadata.get("title") if metadata else "Custom SRS",
                "domain": metadata.get("domain") if metadata else "General",
                "total_requirements": 0, "total_test_cases": 0, "approved_test_cases": 0,
            }
            self.save_results(state)
            return state
        logger.info(f"✅ Target is up: HTTP {health.get('status')}, {health.get('load_ms')} ms, "
                    f"health score {health.get('score')}/100, {len(state['target_profile'].get('inputs', []))} input field(s) found")

        # ----------------------------------------------------
        # Stage 1: Requirement Extraction
        # ----------------------------------------------------
        _update("Stage 1/7: Extracting testable requirements from SRS...", 1)
        state = await self.req_extractor.invoke(state)

        # ----------------------------------------------------
        # Stage 2: Requirement Validation & Ambiguity Detection
        # ----------------------------------------------------
        _update("Stage 2/7: Auditing requirements for clarity, ambiguity, and testability...", 2)
        state = await self.req_validator.invoke(state)

        # ----------------------------------------------------
        # Stage 3: Risk & Coverage Planning
        # ----------------------------------------------------
        _update("Stage 3/7: Analyzing risks and prioritizing automated test suites...", 3)
        state = await self.risk_planner.invoke(state)

        # ----------------------------------------------------
        # Stage 4: Test Case Design (BDD & Gherkin)
        # ----------------------------------------------------
        _update("Stage 4/7: Designing structured test cases and BDD / Gherkin specifications...", 4)
        state = await self.test_designer.invoke(state)

        # ----------------------------------------------------
        # Stage 5: Security Review & OWASP Compliance
        # ----------------------------------------------------
        _update("Stage 5/7: Reviewing test cases against OWASP security compliance rules...", 5)
        state = await self.reviewer.invoke(state)

        # ----------------------------------------------------
        # Stage 6 & 7: Script Generation & Playwright Sandbox Execution
        # ----------------------------------------------------
        _update("Stage 6/7: Generating Playwright automation scripts...", 6)
        
        for attempt in range(max_retries):
            state["retry_count"] = attempt
            _update(f"Stage 7/7: Executing Playwright Script (Attempt #{attempt + 1}/{max_retries})...", 7)

            # Generate or patch script
            state = await self.script_generator.invoke(state)
            current_script = state.get("current_script", "")

            # Execute in sandboxed Playwright engine
            exec_res = self.runner.execute_script(current_script, script_name=f"test_run_v{attempt + 1}.py",
                                                  shot_dir=run_dir)
            state["execution_result"] = exec_res

            # Store per-test-case results (empty list if the run crashed, so stale results never linger)
            state["test_results"] = exec_res.get("test_results") or []

            if exec_res.get("status") == "PASSED":
                logger.info(f"✅ EXECUTION SUCCESSFUL on Attempt #{attempt + 1}!")
                state.setdefault("metrics", {})["passed_scripts"] = state.get("metrics", {}).get("passed_scripts", 0) + 1
                state["human_review_required"] = False
                break
            else:
                logger.warning(f"⚠️ Execution failed on Attempt #{attempt + 1}: {exec_res.get('error')}")
                state.setdefault("metrics", {})["failed_scripts"] = state.get("metrics", {}).get("failed_scripts", 0) + 1
                state["last_error"] = exec_res.get("error")
                
                if "retry_history" not in state or state["retry_history"] is None:
                    state["retry_history"] = []
                    
                state["retry_history"].append({
                    "attempt": attempt + 1,
                    "error": exec_res.get("error"),
                    "duration": exec_res.get("duration_seconds")
                })

                # Always analyze: even on the last attempt it tells the reviewer what failed and why
                logger.info("⚡ Triggering Failure Analysis Agent...")
                state = await self.failure_analyzer.invoke(state)
                analysis = state.get("error_analysis") or {}
                state["retry_history"][-1]["root_cause"] = analysis.get("root_cause")

                if analysis.get("can_auto_recover", True) and attempt < max_retries - 1:
                    logger.info("🔧 Failures look script/locator related - retrying with an adjusted script.")
                    continue

                if attempt < max_retries - 1:
                    logger.info("🛑 Retrying cannot fix these failures - escalating without further attempts.")
                else:
                    logger.error("❌ Max retries reached. Escalating persistent failure to Human Review.")
                state["human_review_required"] = True
                state["human_review_context"] = {
                    "reason": analysis.get("escalation_reason", "persistent_failure"),
                    "last_error": exec_res.get("error"),
                    "total_attempts": attempt + 1,
                    "summary": analysis.get("summary", {}),
                    "failures": analysis.get("failures", []),
                    "script": current_script,
                    "stdout": exec_res.get("stdout"),
                    "stderr": exec_res.get("stderr"),
                    "screenshots": exec_res.get("screenshots", [])
                }
                state.setdefault("metrics", {})["human_interventions"] = 1
                break

        # Build requirement → test → execution traceability
        state["traceability"] = build_traceability(
            state.get("requirements", []),
            state.get("test_cases", []),
            state.get("test_results", [])
        )

        state["llm_trace"] = self._llm_trace()

        state["final_results"] = {
            "completed_at": datetime.now().isoformat(),
            "status": "PASSED" if not state.get("human_review_required") else "REQUIRES_HUMAN_REVIEW",
            "passed": state.get("execution_result", {}).get("status") == "PASSED",
            "document_title": metadata.get("title") if metadata else "Custom SRS",
            "domain": metadata.get("domain") if metadata else "General",
            "total_requirements": len(state.get("requirements", [])),
            "testability_score": state.get("quality_metrics", {}).get("testability_score", 85),
            "total_test_cases": len(state.get("test_cases", [])),
            "approved_test_cases": len(state.get("approved_test_cases", []))
        }

        self.save_results(state)
        logger.info("============================================================")
        logger.info(f"🏁 PIPELINE RUN COMPLETED. Final Status: {state['final_results']['status']}")
        logger.info("============================================================")
        return state

    def _llm_trace(self):
        """Per-stage summary of which LLM was assigned and which one actually answered."""
        agents = [self.req_extractor, self.req_validator, self.risk_planner, self.test_designer,
                  self.reviewer, self.failure_analyzer]
        rows = []
        for agent in agents:
            calls = agent.call_log
            if calls:
                planned = calls[0]["planned"]
            else:
                provider, model = config.get_stage_route(agent.name)
                planned = f"{provider}/{model}"
            used = {}
            for c in calls:
                key = f"{c['provider']}/{c['model']}"
                used[key] = used.get(key, 0) + 1
            errors = [e for c in calls for e in c.get("errors", [])]
            rows.append({
                "stage": agent.name,
                "assigned": planned,
                "answered_by": ", ".join(f"{k} x{n}" for k, n in used.items()) if used else "not needed",
                "calls": len(calls),
                "avg_seconds": round(sum(c["seconds"] for c in calls) / len(calls), 1) if calls else 0.0,
                "fell_back": any(c["fallback"] for c in calls),
                "first_error": errors[0][:140] if errors else "",
            })
        return rows

    def save_results(self, state: AgentState):
        """Persist execution run data to data/pipeline_results.json"""
        out_path = os.path.join(config.DATA_DIR, "pipeline_results.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"💾 Execution results saved to '{out_path}'")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")

async def run_pipeline(srs_text: str, target_url: str = "", metadata: Optional[Dict[str, Any]] = None) -> AgentState:
    pipeline = AgenticQAPipeline()
    return await pipeline.run(srs_text, target_url, metadata=metadata)

if __name__ == "__main__":
    from config import config
    default_srs = config.get_default_srs_path()
    with open(default_srs, "r", encoding="utf-8") as f:
        content = f.read()
    asyncio.run(run_pipeline(content, metadata={"title": "Default SRS", "domain": "Security"}))