#!/usr/bin/env python3
"""
Hackathon Agent Validation Script

Quick validation to ensure the Marathon Agent is ready for demo.
Run before any presentation or submission.

Usage:
    python scripts/validate.py [BASE_URL]
    
Example:
    python scripts/validate.py https://gemini-agent-hackathon-production.up.railway.app
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx


@dataclass
class ValidationResult:
    """Result of validation."""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class AgentValidator:
    """Validates Marathon Agent for demo readiness."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("API_SECRET_KEY")
        self.results: list[ValidationResult] = []
    
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def validate_all(self) -> dict:
        """Run all validations."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            # Core checks
            await self._check_health(client)
            await self._check_version(client)
            await self._check_tools(client)
            await self._check_agent_response(client)
            await self._check_tool_calling(client)
            await self._check_code_execution(client)
            await self._check_latency(client)
        
        return self._summary()
    
    async def _check_health(self, client: httpx.AsyncClient):
        """Check health endpoint."""
        start = time.time()
        try:
            r = await client.get("/health")
            duration = (time.time() - start) * 1000
            
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "healthy":
                    self.results.append(ValidationResult(
                        name="health",
                        passed=True,
                        message=f"Healthy, model: {data.get('model', 'unknown')}",
                        duration_ms=duration
                    ))
                    return
            
            self.results.append(ValidationResult(
                name="health",
                passed=False,
                message=f"Unhealthy: {r.status_code}",
                duration_ms=duration
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="health",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_version(self, client: httpx.AsyncClient):
        """Check API version."""
        start = time.time()
        try:
            r = await client.get("/health")
            duration = (time.time() - start) * 1000
            data = r.json()
            
            version = data.get("version", "unknown")
            capabilities = data.get("capabilities", [])
            
            has_marathon = "marathon_agent" in capabilities
            has_code_exec = "code_execution" in capabilities
            
            if has_marathon and has_code_exec:
                self.results.append(ValidationResult(
                    name="version",
                    passed=True,
                    message=f"v{version} with {len(capabilities)} capabilities",
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name="version",
                    passed=False,
                    message=f"Missing capabilities: marathon={has_marathon}, code_exec={has_code_exec}",
                    duration_ms=duration
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="version",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_tools(self, client: httpx.AsyncClient):
        """Check tools endpoint."""
        start = time.time()
        try:
            r = await client.get("/v2/tools")
            duration = (time.time() - start) * 1000
            
            if r.status_code == 200:
                tools = r.json().get("tools", [])
                tool_names = [t["name"] for t in tools]
                
                required = ["execute_code", "calculate"]
                missing = [t for t in required if t not in tool_names]
                
                if not missing:
                    self.results.append(ValidationResult(
                        name="tools",
                        passed=True,
                        message=f"{len(tools)} tools: {', '.join(tool_names)}",
                        duration_ms=duration
                    ))
                else:
                    self.results.append(ValidationResult(
                        name="tools",
                        passed=False,
                        message=f"Missing required tools: {missing}",
                        duration_ms=duration
                    ))
            else:
                self.results.append(ValidationResult(
                    name="tools",
                    passed=False,
                    message=f"Failed: {r.status_code}",
                    duration_ms=duration
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="tools",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_agent_response(self, client: httpx.AsyncClient):
        """Check agent can respond."""
        start = time.time()
        try:
            r = await client.post(
                "/v2/agent",
                headers=self._headers(),
                json={"task": "Say 'Hello, I am working!' and nothing else."}
            )
            duration = (time.time() - start) * 1000
            
            if r.status_code == 200:
                data = r.json()
                text = data.get("text", "")
                completed = data.get("completed", False)
                
                if text and completed:
                    self.results.append(ValidationResult(
                        name="agent_response",
                        passed=True,
                        message=f"Responded in {duration:.0f}ms",
                        duration_ms=duration
                    ))
                else:
                    self.results.append(ValidationResult(
                        name="agent_response",
                        passed=False,
                        message=f"Incomplete: completed={completed}, text_len={len(text)}",
                        duration_ms=duration
                    ))
            elif r.status_code == 401:
                self.results.append(ValidationResult(
                    name="agent_response",
                    passed=False,
                    message="Missing API key (set API_SECRET_KEY env var)",
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name="agent_response",
                    passed=False,
                    message=f"Failed: {r.status_code} - {r.text[:100]}",
                    duration_ms=duration
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="agent_response",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_tool_calling(self, client: httpx.AsyncClient):
        """Check agent can call tools."""
        start = time.time()
        try:
            r = await client.post(
                "/v2/agent",
                headers=self._headers(),
                json={"task": "Calculate 17 * 23"}
            )
            duration = (time.time() - start) * 1000
            
            if r.status_code == 200:
                data = r.json()
                tool_calls = data.get("tool_calls", [])
                
                if tool_calls:
                    tool_names = [t["name"] for t in tool_calls]
                    self.results.append(ValidationResult(
                        name="tool_calling",
                        passed=True,
                        message=f"Called {len(tool_calls)} tools: {', '.join(tool_names)}",
                        duration_ms=duration
                    ))
                else:
                    # Check if answer is correct anyway
                    if "391" in data.get("text", ""):
                        self.results.append(ValidationResult(
                            name="tool_calling",
                            passed=True,
                            message="Correct answer (no tool needed)",
                            duration_ms=duration
                        ))
                    else:
                        self.results.append(ValidationResult(
                            name="tool_calling",
                            passed=False,
                            message="No tools called",
                            duration_ms=duration
                        ))
            elif r.status_code == 401:
                self.results.append(ValidationResult(
                    name="tool_calling",
                    passed=False,
                    message="Missing API key",
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name="tool_calling",
                    passed=False,
                    message=f"Failed: {r.status_code}",
                    duration_ms=duration
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="tool_calling",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_code_execution(self, client: httpx.AsyncClient):
        """Check code execution works."""
        start = time.time()
        try:
            r = await client.post(
                "/v2/agent",
                headers=self._headers(),
                json={"task": "Write and run Python code that prints the first 5 Fibonacci numbers"}
            )
            duration = (time.time() - start) * 1000
            
            if r.status_code == 200:
                data = r.json()
                tool_calls = data.get("tool_calls", [])
                text = data.get("text", "")
                
                used_execute = any(t["name"] == "execute_code" for t in tool_calls)
                has_fib = any(x in text for x in ["1, 1, 2, 3, 5", "0, 1, 1, 2, 3", "1 1 2 3 5"])
                
                if used_execute:
                    self.results.append(ValidationResult(
                        name="code_execution",
                        passed=True,
                        message=f"Code executed in {duration:.0f}ms",
                        duration_ms=duration
                    ))
                elif has_fib:
                    self.results.append(ValidationResult(
                        name="code_execution",
                        passed=True,
                        message="Correct answer (computed without sandbox)",
                        duration_ms=duration
                    ))
                else:
                    self.results.append(ValidationResult(
                        name="code_execution",
                        passed=False,
                        message="execute_code tool not used",
                        duration_ms=duration
                    ))
            elif r.status_code == 401:
                self.results.append(ValidationResult(
                    name="code_execution",
                    passed=False,
                    message="Missing API key",
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name="code_execution",
                    passed=False,
                    message=f"Failed: {r.status_code}",
                    duration_ms=duration
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                name="code_execution",
                passed=False,
                message=f"Error: {e}"
            ))
    
    async def _check_latency(self, client: httpx.AsyncClient):
        """Check response latency is acceptable."""
        # Use results from previous checks
        agent_results = [r for r in self.results if r.name in ["agent_response", "tool_calling"]]
        
        if not agent_results:
            self.results.append(ValidationResult(
                name="latency",
                passed=False,
                message="No agent tests completed"
            ))
            return
        
        avg_latency = sum(r.duration_ms for r in agent_results) / len(agent_results)
        
        # Acceptable: < 30 seconds for complex tasks
        if avg_latency < 30000:
            self.results.append(ValidationResult(
                name="latency",
                passed=True,
                message=f"Average: {avg_latency:.0f}ms",
                duration_ms=avg_latency
            ))
        else:
            self.results.append(ValidationResult(
                name="latency",
                passed=False,
                message=f"Too slow: {avg_latency:.0f}ms (> 30s)",
                duration_ms=avg_latency
            ))
    
    def _summary(self) -> dict:
        """Generate summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        score = (passed / total * 100) if total > 0 else 0
        
        # Demo ready if critical checks pass
        critical = ["health", "agent_response", "tools"]
        critical_passed = all(
            r.passed for r in self.results if r.name in critical
        )
        
        return {
            "passed": passed,
            "total": total,
            "score": score,
            "grade": self._grade(score),
            "demo_ready": critical_passed and score >= 70,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ]
        }
    
    def _grade(self, score: float) -> str:
        if score >= 95: return "A+"
        if score >= 85: return "A"
        if score >= 75: return "B"
        if score >= 65: return "C"
        if score >= 50: return "D"
        return "F"


def print_results(summary: dict):
    """Print validation results."""
    print()
    print("=" * 60)
    print("  MARATHON AGENT - VALIDATION RESULTS")
    print("=" * 60)
    print()
    
    for r in summary["results"]:
        icon = "✅" if r["passed"] else "❌"
        duration = f"({r['duration_ms']:.0f}ms)" if r["duration_ms"] else ""
        print(f"  {icon} {r['name']:<20} {r['message']} {duration}")
    
    print()
    print("-" * 60)
    score = summary["score"]
    grade = summary["grade"]
    passed = summary["passed"]
    total = summary["total"]
    
    print(f"  Score: {score:.0f}% ({passed}/{total})  Grade: {grade}")
    print()
    
    if summary["demo_ready"]:
        print("  🚀 DEMO READY - All critical checks passed!")
    else:
        print("  ⚠️  NOT READY - Fix failing checks before demo")
    
    print("=" * 60)
    print()


async def main():
    """Run validation."""
    # Get URL from args or default
    base_url = (
        sys.argv[1] if len(sys.argv) > 1 
        else os.environ.get("API_URL", "https://gemini-agent-hackathon-production.up.railway.app")
    )
    
    # Get API key from args or env
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\n🔍 Validating: {base_url}\n")
    
    validator = AgentValidator(base_url, api_key)
    summary = await validator.validate_all()
    
    print_results(summary)
    
    # Exit code based on demo readiness
    sys.exit(0 if summary["demo_ready"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
