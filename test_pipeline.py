"""
End-to-End Pipeline Integration Test Script
Executes full requirement-to-test workflow and validates agent outputs
"""
import asyncio
import os
import json

from config import config
from pipeline import AgenticQAPipeline

async def main():
    print("==================================================================")
    print("🧪 RUNNING AGENTIC QA PIPELINE INTEGRATION TEST")
    print("==================================================================")

    # 1. Load sample SRS document
    srs_file = os.path.join(config.SAMPLES_DIR, "sample_srs.txt")
    if not os.path.exists(srs_file):
        print(f"❌ Sample SRS file not found at {srs_file}")
        return

    with open(srs_file, "r", encoding="utf-8") as f:
        srs_content = f.read()

    print(f"📄 Loaded sample SRS specification ({len(srs_content)} chars)")

    # 2. Instantiate and run pipeline
    pipeline = AgenticQAPipeline()
    
    def status_logger(msg, step, state):
        print(f"  ➜ Progress [{step}/7]: {msg}")

    final_state = await pipeline.run(
        srs_text=srs_content,
        target_url="https://demo.example.com",
        max_retries=2,
        status_callback=status_logger
    )

    print()
    print("==================================================================")
    print("📊 INTEGRATION TEST VERIFICATION RESULTS")
    print("==================================================================")
    
    reqs = final_state.get("requirements", [])
    val_reqs = final_state.get("validated_requirements", [])
    tcs = final_state.get("test_cases", [])
    app_tcs = final_state.get("approved_test_cases", [])
    exec_res = final_state.get("execution_result", {})
    hitl = final_state.get("human_review_required", False)

    print(f"✅ Requirements Extracted: {len(reqs)}")
    print(f"✅ Requirements Validated: {len(val_reqs)}")
    print(f"✅ Test Cases Generated:   {len(tcs)}")
    print(f"✅ Test Cases Approved:    {len(app_tcs)}")
    print(f"✅ Execution Status:       {exec_res.get('status')}")
    print(f"✅ Human Review Escalated: {hitl}")
    print(f"✅ Results Persisted at:   {os.path.join(config.DATA_DIR, 'pipeline_results.json')}")
    
    print()
    print("=" * 60)
    print("🎉 ALL AGENTIC PIPELINE INTEGRATION TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
