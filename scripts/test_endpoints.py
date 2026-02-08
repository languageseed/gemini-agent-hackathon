#!/usr/bin/env python3
"""
Quick endpoint tests - run before deploying.

Usage:
    # Test against local server
    python scripts/test_endpoints.py
    
    # Test against production
    python scripts/test_endpoints.py --prod
"""

import os
import sys
import argparse
import requests
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--prod", action="store_true", help="Test production server")
    parser.add_argument("--api-key", default=os.environ.get("API_SECRET_KEY", ""), help="API key")
    args = parser.parse_args()
    
    if args.prod:
        base_url = "https://gemini-agent-hackathon-production.up.railway.app"
    else:
        base_url = "http://localhost:8000"
    
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    
    print(f"🔍 Testing: {base_url}")
    print("=" * 50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health check
    print("\n1️⃣ Health Check (GET /health)")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {resp.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 2: Public status endpoint
    print("\n2️⃣ Public Status (GET /status)")
    try:
        resp = requests.get(f"{base_url}/status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Uptime: {data.get('uptime_seconds', 0):.0f}s")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {resp.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 3: Root endpoint
    print("\n3️⃣ Root (GET /)")
    try:
        resp = requests.get(f"{base_url}/", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            endpoints = data.get("endpoints", {})
            print(f"   ✅ Available endpoints: {len(endpoints)}")
            tests_passed += 1
        elif resp.status_code == 401:
            print(f"   ⚠️  Requires API key (401) - endpoint exists")
            tests_passed += 1  # Expected if API key required
        else:
            print(f"   ❌ Status: {resp.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Test 4: Async job submission (dry run)
    print("\n4️⃣ Async Job API (POST /v4/analyze/async)")
    try:
        resp = requests.post(
            f"{base_url}/v4/analyze/async",
            headers=headers,
            json={"repo_url": "test", "focus": "security"},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            job_id = data.get("job_id")
            print(f"   ✅ Job created: {job_id}")
            tests_passed += 1
            
            # Check job status
            print("\n5️⃣ Job Status (GET /v4/jobs/{job_id})")
            status_resp = requests.get(f"{base_url}/v4/jobs/{job_id}", headers=headers, timeout=10)
            if status_resp.status_code == 200:
                status = status_resp.json()
                print(f"   ✅ Job status: {status.get('status')}")
                tests_passed += 1
            else:
                print(f"   ❌ Status: {status_resp.status_code}")
                tests_failed += 1
        elif resp.status_code == 401:
            print(f"   ⚠️  Requires API key (401)")
            tests_passed += 1
        elif resp.status_code == 422:
            print(f"   ⚠️  Validation error (expected without valid repo)")
            tests_passed += 1
        else:
            print(f"   ❌ Status: {resp.status_code}")
            print(f"   Response: {resp.text[:200]}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Results: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed > 0:
        sys.exit(1)
    print("✅ All tests passed!")


if __name__ == "__main__":
    main()
