#!/usr/bin/env python3
"""
Local Code Analysis Script

Run the verified analyzer locally on the codebase before deploying.
This catches bugs before they go to production.

Usage:
    # Analyze this repo
    python scripts/analyze_local.py
    
    # Analyze specific path
    python scripts/analyze_local.py --path src/
    
    # Quick mode (no verification)
    python scripts/analyze_local.py --quick
    
    # Focus on security only
    python scripts/analyze_local.py --focus security

Requirements:
    - GOOGLE_API_KEY environment variable set
    - E2B_API_KEY for verification (optional, will skip if not set)
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Run local code analysis")
    parser.add_argument("--path", default="src/", help="Path to analyze (default: src/)")
    parser.add_argument("--focus", default="all", choices=["all", "bugs", "security", "performance", "architecture"])
    parser.add_argument("--quick", action="store_true", help="Skip verification (faster)")
    parser.add_argument("--max-issues", type=int, default=10, help="Max issues to verify")
    args = parser.parse_args()
    
    # Check for required env vars
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Get your API key from: https://aistudio.google.com/apikey")
        sys.exit(1)
    
    if not args.quick and not os.environ.get("E2B_API_KEY"):
        print("⚠️  Warning: E2B_API_KEY not set - verification will be limited")
        print("   Get your API key from: https://e2b.dev/dashboard")
    
    asyncio.run(run_analysis(args))


async def run_analysis(args):
    from google import genai
    from agent.verified_analysis import VerifiedAnalyzer
    from agent.stream import EventType
    
    print("=" * 60)
    print("🔍 LOCAL CODE ANALYSIS")
    print("=" * 60)
    print(f"Path: {args.path}")
    print(f"Focus: {args.focus}")
    print(f"Verify: {not args.quick}")
    print(f"Max Issues: {args.max_issues}")
    print("=" * 60)
    
    # Collect codebase content
    print("\n📂 Collecting codebase...")
    repo_content = collect_codebase(args.path)
    print(f"   Collected {len(repo_content):,} characters from {args.path}")
    
    # Initialize Gemini client (new google-genai SDK)
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Use the reasoning model (same as production) for deep analysis
    model_name = os.environ.get("GEMINI_REASONING_MODEL", "gemini-3-pro-preview")
    print(f"   Using model: {model_name}")
    
    # Initialize analyzer with client
    analyzer = VerifiedAnalyzer(client=client, model=model_name)
    
    # Track events
    issues_found = []
    verified_count = 0
    
    def on_event(event):
        nonlocal verified_count, issues_found
        if event.type == EventType.TOOL_RESULT:
            name = event.data.get("name", "")
            if name == "issue_found":
                # Issue discovered during analysis
                issues_found.append(event.data)
                severity = event.data.get("severity", "unknown").upper()
                title = event.data.get("title", "Unknown")
                print(f"   🔍 [{severity}] {title}")
            elif name == "verify_issue":
                status = event.data.get("status")
                if status == "verified":
                    verified_count += 1
                    print(f"   ✅ VERIFIED: {event.data.get('issue_id', '')[:8]}")
                elif status == "unverified":
                    print(f"   ⚠️  Unverified: {event.data.get('issue_id', '')[:8]}")
                elif status == "error":
                    print(f"   ❌ Error: {event.data.get('message', '')[:50]}")
            elif name == "generate_fix":
                status = event.data.get("status")
                if status == "proposed":
                    print(f"   🔧 Fix proposed: {event.data.get('issue_id', '')[:8]}")
        elif event.type == EventType.THINKING:
            phase = event.data.get("phase", "")
            message = event.data.get("message", "")
            if phase:
                print(f"\n📍 Phase: {phase}")
            if message:
                print(f"   {message}")
    
    # Run analysis
    print("\n🤖 Running Gemini analysis...")
    start_time = datetime.now()
    
    try:
        result = await analyzer.analyze_and_verify(
            repo_content=repo_content,
            repo_url="local://codebase",
            focus=args.focus,
            on_event=on_event,
            max_issues_to_verify=0 if args.quick else args.max_issues,
        )
    except Exception as e:
        import traceback
        print(f"\n❌ Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print(f"Issues Found: {result.total_issues}")
    print(f"Verified: {result.verified_count}")
    print(f"Unverified: {result.unverified_count}")
    
    # Print issues by severity
    if result.issues:
        print("\n📋 Issues by Severity:")
        for severity in ["critical", "high", "medium", "low"]:
            count = sum(1 for i in result.issues if i.severity.value == severity)
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(severity, "")
                print(f"   {emoji} {severity.upper()}: {count}")
        
        print("\n📝 Issue Details:")
        for issue in result.issues:
            status = "✓" if issue.verification_status.value == "verified" else "?"
            print(f"   [{status}] [{issue.severity.value.upper()}] {issue.title}")
            print(f"       File: {issue.file_path}")
    
    # Exit code based on critical/high issues
    critical_high = sum(1 for i in result.issues 
                       if i.severity.value in ["critical", "high"] 
                       and i.verification_status.value == "verified")
    
    if critical_high > 0:
        print(f"\n⚠️  Found {critical_high} verified critical/high issues!")
        sys.exit(1)
    else:
        print("\n✅ No verified critical/high issues found")
        sys.exit(0)


def collect_codebase(path: str) -> str:
    """Collect codebase content in the format expected by the analyzer."""
    base_path = Path(__file__).parent.parent / path
    
    if not base_path.exists():
        print(f"❌ Path not found: {base_path}")
        sys.exit(1)
    
    # Extensions to include
    extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}
    
    # Directories to skip
    skip_dirs = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"}
    
    content_parts = []
    file_count = 0
    
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            # Skip excluded directories
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            
            # Check extension
            if file_path.suffix.lower() not in extensions:
                continue
            
            try:
                relative_path = file_path.relative_to(base_path.parent)
                file_content = file_path.read_text(encoding="utf-8", errors="replace")
                
                # Format like the GitHub cloner does
                content_parts.append(f"### {relative_path}\n```{get_language(file_path)}\n{file_content}\n```\n")
                file_count += 1
            except Exception as e:
                print(f"   Warning: Could not read {file_path}: {e}")
    
    print(f"   Collected {file_count} files")
    return "\n".join(content_parts)


def get_language(path: Path) -> str:
    """Get language name from file extension."""
    mapping = {
        ".py": "python",
        ".js": "javascript", 
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
    }
    return mapping.get(path.suffix.lower(), "")


if __name__ == "__main__":
    main()
