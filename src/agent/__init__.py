"""
Marathon Agent Module

Minimal, powerful agent leveraging Gemini 3 native capabilities.

Features:
- Code Analysis with Verification (Vibe Engineering)
- Security Scanning (Secret detection, container checks)
- Evolution Advisor (Roadmap generation, tech debt analysis)
"""

from .core import MarathonAgent, AgentConfig, AgentResult
from .tools import ToolRegistry, ToolResult, execute_code_in_sandbox, execute_verification_test, VerificationResult, classify_test_result
from .session import SessionStore, Session
from .stream import StreamEvent, EventType, EventCollector
from .verified_analysis import (
    VerifiedAnalyzer,
    Issue,
    AnalysisResult,
    Severity,
    Category,
    IssueType,
    VerificationStatus,
)
from .security import (
    SecretScanner,
    SecurityFinding,
    SecurityScanResult,
    scan_codebase_for_secrets,
    scan_codebase_for_secrets_async,
    Severity as SecuritySeverity,
    FindingCategory,
    SecretType,
)
from .evolution import (
    EvolutionAdvisor,
    EvolutionReport,
    EvolutionRecommendation,
    EvolutionCategory,
    Priority,
    Effort,
)

__all__ = [
    "MarathonAgent",
    "AgentConfig", 
    "AgentResult",
    "ToolRegistry",
    "ToolResult",
    "execute_code_in_sandbox",
    "SessionStore",
    "Session",
    "StreamEvent",
    "EventType",
    "EventCollector",
    # Verified Analysis
    "VerifiedAnalyzer",
    "Issue",
    "AnalysisResult",
    "Severity",
    "Category",
    "VerificationStatus",
    # Security Scanning
    "SecretScanner",
    "SecurityFinding",
    "SecurityScanResult",
    "scan_codebase_for_secrets",
    "scan_codebase_for_secrets_async",
    "SecuritySeverity",
    "FindingCategory",
    "SecretType",
    # Evolution Advisor
    "EvolutionAdvisor",
    "EvolutionReport",
    "EvolutionRecommendation",
    "EvolutionCategory",
    "Priority",
    "Effort",
]
