"""
Command Line Interface (CLI) Entrypoint for CI/CD Pipeline Integration
Enhanced with PURE Benchmark Dataset Support
Usage:
  python run_cli.py --list-pure
  python run_cli.py --pure keepass --url http://127.0.0.1:8080
  python run_cli.py --srs samples/pure/2008_keepass_srs.txt --retries 2
"""
import argparse
import asyncio
import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import config
from pipeline import AgenticQAPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic QA Framework CLI Runner (PURE Dataset Enabled)")
    parser.add_argument("--srs", type=str, default="", help="Path to SRS document file")
    parser.add_argument("--pure", type=str, default="", help="ID or keyword of PURE dataset document to run (e.g., keepass, cctns, themas, peppol)")
    parser.add_argument("--list-pure", action="store_true", help="List all available PURE dataset documents and exit")
    parser.add_argument("--url", type=str, default=config.TARGET_URL, help="Target application URL")
    parser.add_argument("--retries", type=int, default=config.MAX_RETRIES, help="Maximum automated retries")
    return parser.parse_args()

def list_pure_documents():
    catalog = config.get_pure_catalog()
    print("=" * 80)
    print(f"📚 AVAILABLE PURE BENCHMARK SPECIFICATIONS ({len(catalog)} Documents)")
    print("=" * 80)
    print(f"{'ID':<25} {'DOMAIN':<35} {'REQS':<8} {'TITLE'}")
    print("-" * 80)
    for doc in catalog:
        print(f"{doc['id']:<25} {doc['domain'][:33]:<35} {doc['requirements_count']:<8} {doc['title'][:40]}")
    print("=" * 80)
    print("Run with: python run_cli.py --pure <ID>")

async def main():
    args = parse_args()

    if args.list_pure:
        list_pure_documents()
        sys.exit(0)

    srs_content = ""
    metadata = None

    if args.pure:
        catalog = config.get_pure_catalog()
        query = args.pure.lower().strip()
        selected_doc = None
        for doc in catalog:
            if (query == doc["id"].lower() or 
                query in doc["filename"].lower() or 
                query in doc["title"].lower()):
                selected_doc = doc
                break

        if not selected_doc:
            print(f"❌ Could not find PURE document matching: '{args.pure}'")
            print("Run 'python run_cli.py --list-pure' to see all available documents.")
            sys.exit(1)

        doc_path = selected_doc.get("path") or os.path.join(config.PURE_SAMPLES_DIR, selected_doc["filename"])
        if not os.path.exists(doc_path):
            print(f"❌ File not found on disk: {doc_path}")
            sys.exit(1)

        with open(doc_path, "r", encoding="utf-8") as f:
            srs_content = f.read()
        metadata = selected_doc
        print(f"📄 Loaded PURE Document: {selected_doc['title']} ({selected_doc['domain']})")
    
    elif args.srs:
        if not os.path.exists(args.srs):
            print(f"❌ SRS file not found: {args.srs}")
            sys.exit(1)
        with open(args.srs, "r", encoding="utf-8") as f:
            srs_content = f.read()
        metadata = {"title": os.path.basename(args.srs), "domain": "User Specified"}
    
    else:
        # Default to preferred PURE document
        default_path = config.get_default_srs_path()
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                srs_content = f.read()
            metadata = {"title": os.path.basename(default_path), "domain": "PURE Benchmark"}
            print(f"📄 Using default specification: {default_path}")
        else:
            print("❌ No SRS document specified and default not found.")
            sys.exit(1)

    print(f"🚀 Launching Agentic QA Pipeline CLI against: {args.url}")
    print(f"Document Size: {len(srs_content):,} characters")
    print("-" * 60)

    pipeline = AgenticQAPipeline()
    state = await pipeline.run(
        srs_text=srs_content,
        target_url=args.url,
        max_retries=args.retries,
        metadata=metadata
    )

    status = state.get("final_results", {}).get("status", "UNKNOWN")
    print("-" * 60)
    print(f"🏁 Pipeline Finished. Final Status: {status}")
    print(f"   Requirements Extracted: {len(state.get('requirements', []))}")
    print(f"   Testability Index:      {state.get('quality_metrics', {}).get('testability_score', 'N/A')}%")
    print(f"   Test Cases Designed:    {len(state.get('test_cases', []))}")
    print(f"   Approved for Execution: {len(state.get('approved_test_cases', []))}")
    print(f"   Execution Result:       {state.get('execution_result', {}).get('status', 'N/A')}")
    
    if state.get("human_review_required"):
        print("⚠️ Human review required for persistent failures.")
        sys.exit(2)
    else:
        print("✅ Automated QA Pipeline verification completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
