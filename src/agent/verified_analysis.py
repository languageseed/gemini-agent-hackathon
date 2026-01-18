"""
Verified Codebase Analysis

Two-phase analysis that:
1. Analyzes codebase for issues
2. Generates and runs tests to VERIFY findings

This is the "Vibe Engineering" approach - agents that verify their work.
"""

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from enum import Enum

import structlog

from .stream import StreamEvent, EventType

logger = structlog.get_logger()


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    ARCHITECTURE = "architecture"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"      # Test failed as expected = bug confirmed
    UNVERIFIED = "unverified"  # Test passed = may be false positive
    SKIPPED = "skipped"        # Could not generate/run test
    ERROR = "error"            # Test execution failed


@dataclass
class Issue:
    """A detected issue in the codebase."""
    id: str
    title: str
    description: str
    severity: Severity
    category: Category
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    recommended_code: Optional[str] = None
    
    # Verification fields
    verification_status: VerificationStatus = VerificationStatus.PENDING
    test_code: Optional[str] = None
    test_output: Optional[str] = None
    test_error: Optional[str] = None
    
    # Fix fields (for verified issues)
    fix_code: Optional[str] = None
    fix_description: Optional[str] = None
    fix_status: str = "none"  # none, proposed, verified, failed
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "recommended_code": self.recommended_code,
            "verification_status": self.verification_status.value,
            "test_code": self.test_code,
            "test_output": self.test_output,
            "test_error": self.test_error,
            "fix_code": self.fix_code,
            "fix_description": self.fix_description,
            "fix_status": self.fix_status,  # none, proposed, verified, failed
        }


@dataclass
class AnalysisResult:
    """Complete analysis result with verification."""
    repo_url: str
    focus: str
    issues: list[Issue]
    summary: str
    files_analyzed: int
    total_issues: int
    verified_count: int
    unverified_count: int
    analysis_time_seconds: float
    verification_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "focus": self.focus,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "files_analyzed": self.files_analyzed,
            "total_issues": self.total_issues,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "analysis_time_seconds": self.analysis_time_seconds,
            "verification_time_seconds": self.verification_time_seconds,
        }


# Prompt to extract structured issues from analysis
ISSUE_EXTRACTION_PROMPT = '''You are analyzing code issues. Extract ONLY the issues from the analysis as a JSON array.

For each issue, provide:
- title: Short title (max 60 chars)
- description: Detailed description
- severity: "critical", "high", "medium", or "low"
- category: "bug", "security", "performance", "style", or "architecture"
- file_path: Path to the file (e.g., "src/api/users.py")
- line_number: Line number if known, null otherwise
- code_snippet: The problematic code (max 10 lines)
- recommendation: How to fix it
- recommended_code: Fixed code example

Return ONLY a JSON array, no markdown, no explanation:
[
  {
    "title": "...",
    "description": "...",
    "severity": "...",
    "category": "...",
    "file_path": "...",
    "line_number": null,
    "code_snippet": "...",
    "recommendation": "...",
    "recommended_code": "..."
  }
]

Analysis to extract from:
'''

# Prompt to generate verification tests
TEST_GENERATION_PROMPT = '''Generate a SELF-CONTAINED Python test that verifies this bug exists.

Issue: {title}
Description: {description}
File: {file_path}
Problematic code:
```
{code_snippet}
```

CRITICAL REQUIREMENTS:
1. The test MUST be completely self-contained - embed/copy the problematic code directly in the test
2. DO NOT import from the repository - the test runs in an isolated sandbox
3. Only use Python standard library (no pip packages)
4. Use assert statements that will FAIL if the bug exists
5. The test should PASS when the bug is fixed

Example structure:
```python
# Test for: {title}
# Copy the problematic function/code here
def buggy_function():
    # paste the code with the bug
    pass

# Test that demonstrates the bug
def test_bug():
    result = buggy_function()
    assert result == expected, f"Bug exists: got {{result}}, expected {{expected}}"

# Run test
test_bug()
print("PASS: Bug was fixed")
```

Return ONLY the Python test code, no markdown or explanation:
'''

# Prompt to generate fix code
FIX_GENERATION_PROMPT = '''Generate a fix for this VERIFIED bug.

Issue: {title}
Description: {description}
File: {file_path}
Line: {line_number}

Original problematic code:
```
{code_snippet}
```

The bug was verified by this test (which FAILED on the original code):
```python
{test_code}
```

Test output showing the failure:
{test_output}

Requirements:
1. Provide the CORRECTED code that would make the test PASS
2. Keep the fix minimal - only change what's necessary
3. Preserve the original code style and structure
4. Explain what you changed and why

Return your response in this JSON format:
{{
  "fix_code": "the corrected code snippet",
  "fix_description": "brief explanation of what was changed and why"
}}
'''


def parse_issues_from_json(json_str: str) -> list[Issue]:
    """Parse issues from JSON string."""
    # Try to extract JSON from potential markdown
    json_match = re.search(r'\[[\s\S]*\]', json_str)
    if json_match:
        json_str = json_match.group(0)
    
    try:
        data = json.loads(json_str)
        issues = []
        
        for item in data:
            try:
                issue = Issue(
                    id=str(uuid.uuid4())[:8],
                    title=item.get("title", "Unknown Issue"),
                    description=item.get("description", ""),
                    severity=Severity(item.get("severity", "medium").lower()),
                    category=Category(item.get("category", "bug").lower()),
                    file_path=item.get("file_path", "unknown"),
                    line_number=item.get("line_number"),
                    code_snippet=item.get("code_snippet"),
                    recommendation=item.get("recommendation"),
                    recommended_code=item.get("recommended_code"),
                )
                issues.append(issue)
            except (ValueError, KeyError) as e:
                logger.warning("issue_parse_error", error=str(e), item=item)
                continue
        
        return issues
        
    except json.JSONDecodeError as e:
        logger.error("json_parse_error", error=str(e), json_str=json_str[:500])
        return []


def extract_test_code(response: str) -> str:
    """Extract Python code from response, removing markdown if present."""
    # Remove markdown code blocks
    code = re.sub(r'^```python\s*', '', response.strip())
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    return code.strip()


class VerifiedAnalyzer:
    """
    Two-phase analyzer that verifies findings.
    
    Phase 1: Analyze codebase and extract issues
    Phase 2: Generate tests and verify each issue
    """
    
    def __init__(
        self,
        client,  # Gemini client
        model: str = "gemini-2.0-flash",
        code_executor: Optional[Callable[[str], tuple[bool, str]]] = None,
    ):
        self.client = client
        self.model = model
        self.code_executor = code_executor
    
    async def analyze_and_verify(
        self,
        repo_content: str,
        repo_url: str,
        focus: str = "all",
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        max_issues_to_verify: int = 10,
        timeout_seconds: float = 600.0,  # 10 minutes default
    ) -> AnalysisResult:
        """
        Analyze codebase and verify findings.
        
        Args:
            repo_content: Full repository content (from clone_repo tool)
            repo_url: Repository URL for reference
            focus: Analysis focus (bugs, security, performance, all)
            on_event: Callback for streaming events
            max_issues_to_verify: Maximum issues to run verification on
        """
        import time
        import asyncio
        
        start_time = time.time()
        
        def check_timeout():
            """Raise if timeout exceeded."""
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Analysis timeout after {elapsed:.1f}s (limit: {timeout_seconds}s)")
        
        def emit(event_type: EventType, data: dict):
            if on_event:
                on_event(StreamEvent(type=event_type, data=data))
        
        # ============================================================
        # PHASE 1: ANALYSIS
        # ============================================================
        emit(EventType.THINKING, {"phase": "analysis", "message": "Analyzing codebase for issues..."})
        
        analysis_start = time.time()
        
        # Build analysis prompt
        focus_prompts = {
            "bugs": "Focus on finding potential bugs, logic errors, edge cases, and error handling issues.",
            "security": "Focus on security vulnerabilities, injection risks, authentication issues, and data exposure.",
            "performance": "Focus on performance bottlenecks, inefficient algorithms, and resource management.",
            "all": "Perform a comprehensive analysis covering bugs, security, performance, and architecture."
        }
        focus_instruction = focus_prompts.get(focus, focus_prompts["all"])
        
        analysis_prompt = f"""Analyze this codebase and identify issues.

{focus_instruction}

For each issue found, provide:
1. A clear title
2. Detailed description
3. Severity rating (critical, high, medium, low)
4. Category (bug, security, performance, style, architecture)
5. File path and line number if possible
6. The problematic code snippet
7. Recommendation with fixed code example

Be specific and actionable. Prioritize real issues over style nitpicks.

CODEBASE:
{repo_content[:500000]}  # Limit to ~125K tokens
"""
        
        # Get analysis from Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=analysis_prompt,
        )
        
        raw_analysis = response.text
        analysis_time = time.time() - analysis_start
        
        emit(EventType.TOKEN, {"content": raw_analysis, "phase": "analysis"})
        
        check_timeout()  # Check after analysis phase
        
        # ============================================================
        # PHASE 2: EXTRACT STRUCTURED ISSUES
        # ============================================================
        emit(EventType.THINKING, {"phase": "extraction", "message": "Extracting structured issues..."})
        
        extraction_prompt = ISSUE_EXTRACTION_PROMPT + raw_analysis
        
        extraction_response = self.client.models.generate_content(
            model=self.model,
            contents=extraction_prompt,
        )
        
        issues = parse_issues_from_json(extraction_response.text)
        
        # Emit each issue found
        for issue in issues:
            emit(EventType.TOOL_RESULT, {
                "name": "issue_found",
                "issue": issue.to_dict()
            })
        
        emit(EventType.THINKING, {
            "phase": "extraction_complete", 
            "message": f"Found {len(issues)} issues",
            "count": len(issues)
        })
        
        # ============================================================
        # PHASE 3: VERIFICATION
        # ============================================================
        if not self.code_executor:
            # No executor available - skip verification
            for issue in issues:
                issue.verification_status = VerificationStatus.SKIPPED
            
            verification_time = 0.0
        else:
            emit(EventType.THINKING, {"phase": "verification", "message": "Generating verification tests..."})
            
            verification_start = time.time()
            
            # Verify top issues (prioritize by severity)
            severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
            sorted_issues = sorted(issues, key=lambda i: severity_order.get(i.severity, 3))
            
            for issue in sorted_issues[:max_issues_to_verify]:
                check_timeout()  # Check before each verification
                await self._verify_issue(issue, emit)
            
            # Mark remaining as skipped
            for issue in sorted_issues[max_issues_to_verify:]:
                issue.verification_status = VerificationStatus.SKIPPED
            
            verification_time = time.time() - verification_start
        
        # ============================================================
        # PHASE 3.5: FIX GENERATION (for verified issues)
        # ============================================================
        verified_issues = [i for i in issues if i.verification_status == VerificationStatus.VERIFIED]
        
        if verified_issues:
            emit(EventType.THINKING, {"phase": "fix_generation", "message": f"Generating fixes for {len(verified_issues)} verified issues..."})
            
            for issue in verified_issues:
                await self._generate_fix(issue, emit)
        
        # ============================================================
        # PHASE 4: GENERATE SUMMARY
        # ============================================================
        verified_count = sum(1 for i in issues if i.verification_status == VerificationStatus.VERIFIED)
        unverified_count = sum(1 for i in issues if i.verification_status == VerificationStatus.UNVERIFIED)
        
        summary = self._generate_summary(issues, repo_url, focus)
        
        result = AnalysisResult(
            repo_url=repo_url,
            focus=focus,
            issues=issues,
            summary=summary,
            files_analyzed=repo_content.count("### "),  # Rough file count
            total_issues=len(issues),
            verified_count=verified_count,
            unverified_count=unverified_count,
            analysis_time_seconds=analysis_time,
            verification_time_seconds=verification_time,
        )
        
        return result
    
    async def _verify_issue(
        self,
        issue: Issue,
        emit: Callable[[EventType, dict], None],
    ):
        """Generate and run a verification test for an issue."""
        from .tools import execute_verification_test
        
        emit(EventType.TOOL_START, {
            "name": "verify_issue",
            "issue_id": issue.id,
            "issue_title": issue.title,
        })
        
        # Generate test
        test_prompt = TEST_GENERATION_PROMPT.format(
            title=issue.title,
            description=issue.description,
            file_path=issue.file_path,
            code_snippet=issue.code_snippet or "N/A",
        )
        
        try:
            test_response = self.client.models.generate_content(
                model=self.model,
                contents=test_prompt,
            )
            
            test_code = extract_test_code(test_response.text)
            issue.test_code = test_code
            
            # Execute test with proper classification
            result = await execute_verification_test(test_code)
            issue.test_output = result.output
            issue.test_error = result.error_message
            
            if result.is_verified:
                # Test failed due to assertion = bug is VERIFIED
                issue.verification_status = VerificationStatus.VERIFIED
                emit(EventType.TOOL_RESULT, {
                    "name": "verify_issue",
                    "issue_id": issue.id,
                    "status": "verified",
                    "error_type": result.error_type,
                    "message": "Bug confirmed - assertion failed as expected",
                })
            elif result.is_unverified:
                # Test passed = bug may be false positive
                issue.verification_status = VerificationStatus.UNVERIFIED
                emit(EventType.TOOL_RESULT, {
                    "name": "verify_issue",
                    "issue_id": issue.id,
                    "status": "unverified",
                    "message": "Test passed - may be false positive",
                })
            else:
                # Environmental error - cannot determine verification
                issue.verification_status = VerificationStatus.ERROR
                emit(EventType.TOOL_RESULT, {
                    "name": "verify_issue",
                    "issue_id": issue.id,
                    "status": "error",
                    "error_type": result.error_type,
                    "message": f"Cannot verify: {result.error_message}",
                })
                logger.warning(
                    "verification_env_error",
                    issue_id=issue.id,
                    error_type=result.error_type,
                    message=result.error_message,
                )
                
        except Exception as e:
            logger.error("verification_error", issue_id=issue.id, error=str(e))
            issue.verification_status = VerificationStatus.ERROR
            issue.test_error = str(e)
            
            emit(EventType.TOOL_RESULT, {
                "name": "verify_issue",
                "issue_id": issue.id,
                "status": "error",
                "error_type": "Exception",
                "message": str(e),
            })
    
    async def _generate_fix(
        self,
        issue: Issue,
        emit: Callable[[EventType, dict], None],
    ):
        """Generate a fix for a verified issue."""
        
        emit(EventType.TOOL_START, {
            "name": "generate_fix",
            "issue_id": issue.id,
            "issue_title": issue.title,
        })
        
        try:
            fix_prompt = FIX_GENERATION_PROMPT.format(
                title=issue.title,
                description=issue.description,
                file_path=issue.file_path,
                line_number=issue.line_number or "unknown",
                code_snippet=issue.code_snippet or "N/A",
                test_code=issue.test_code or "N/A",
                test_output=issue.test_output or "N/A",
            )
            
            fix_response = self.client.models.generate_content(
                model=self.model,
                contents=fix_prompt,
            )
            
            # Parse the JSON response
            fix_text = fix_response.text
            
            # Extract JSON from potential markdown
            json_match = re.search(r'\{[\s\S]*\}', fix_text)
            if json_match:
                fix_data = json.loads(json_match.group(0))
                issue.fix_code = fix_data.get("fix_code")
                issue.fix_description = fix_data.get("fix_description")
                
                # Mark as proposed - actual verification would require:
                # 1. Creating a test with the fixed code
                # 2. Running the test and confirming it passes
                # For now, we're honest that fixes are proposals
                issue.fix_status = "proposed"
                
                emit(EventType.TOOL_RESULT, {
                    "name": "generate_fix",
                    "issue_id": issue.id,
                    "status": "success",
                    "fix_status": "proposed",  # Be explicit this is a proposal
                    "fix_description": issue.fix_description,
                })
            else:
                # Couldn't parse JSON, use raw response
                issue.fix_description = fix_text[:500]
                issue.fix_status = "proposed"
                emit(EventType.TOOL_RESULT, {
                    "name": "generate_fix",
                    "issue_id": issue.id,
                    "status": "partial",
                    "fix_status": "proposed",
                    "message": "Could not parse structured fix",
                })
                
        except Exception as e:
            logger.error("fix_generation_error", issue_id=issue.id, error=str(e))
            emit(EventType.TOOL_RESULT, {
                "name": "generate_fix",
                "issue_id": issue.id,
                "status": "error",
                "message": str(e),
            })
    
    def _generate_summary(self, issues: list[Issue], repo_url: str, focus: str) -> str:
        """Generate a human-readable summary."""
        
        severity_counts = {s: 0 for s in Severity}
        for issue in issues:
            severity_counts[issue.severity] += 1
        
        verified = sum(1 for i in issues if i.verification_status == VerificationStatus.VERIFIED)
        unverified = sum(1 for i in issues if i.verification_status == VerificationStatus.UNVERIFIED)
        
        summary = f"""## Analysis Summary for {repo_url}

**Focus:** {focus}
**Total Issues:** {len(issues)}
**Verified:** {verified} (confirmed via automated tests)
**Unverified:** {unverified} (may be false positives)

### By Severity
- 🔴 Critical: {severity_counts[Severity.CRITICAL]}
- 🟠 High: {severity_counts[Severity.HIGH]}
- 🟡 Medium: {severity_counts[Severity.MEDIUM]}
- ⚪ Low: {severity_counts[Severity.LOW]}
"""
        
        # Top issues
        if issues:
            summary += "\n### Top Issues\n"
            for issue in sorted(issues, key=lambda i: (
                {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}[i.severity],
                0 if i.verification_status == VerificationStatus.VERIFIED else 1
            ))[:5]:
                status_icon = "✓" if issue.verification_status == VerificationStatus.VERIFIED else "?"
                summary += f"- [{issue.severity.value.upper()}] {status_icon} {issue.title}\n"
        
        return summary
