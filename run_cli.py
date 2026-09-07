"""
Command Line Interface (CLI) Entrypoint for CI/CD Pipeline Integration
Usage: python run_cli.py --srs samples/sample_srs.txt --url http://127.0.0.1:8080 --retries 3
"""
import argparse
import asyncio
import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import config
from pipeline import AgenticQAPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic QA Framework CLI Runner")
    parser.add_argument("--srs", type=str, default=os.path.join(config.SAMPLES_DIR, "sample_srs.txt"), help="Path to SRS document file")
    parser.add_argument("--url", type=str, default=config.TARGET_URL, help="Target application URL")
    parser.add_argument("--retries", type=int, default=config.MAX_RETRIES, help="Maximum automated retries")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    if not os.path.exists(args.srs):
        print(f"❌ SRS file not found: {args.srs}")
        sys.exit(1)

    with open(args.srs, "r", encoding="utf-8") as f:
        srs_text = f.read()

    print(f"🚀 Launching Agentic QA Pipeline CLI against: {args.url}")
    pipeline = AgenticQAPipeline()
    state = await pipeline.run(srs_text=srs_text, target_url=args.url, max_retries=args.retries)

    status = state.get("final_results", {}).get("status", "UNKNOWN")
    print(f"🏁 Pipeline Finished. Status: {status}")
    
    if state.get("human_review_required"):
        print("⚠️ Human review required for persistent failures.")
        sys.exit(2)
    else:
        print("✅ All automated tests executed and verified successfully.")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
