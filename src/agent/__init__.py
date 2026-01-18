"""
Marathon Agent Module

Minimal, powerful agent leveraging Gemini 3 native capabilities.
"""

from .core import MarathonAgent, AgentConfig, AgentResult
from .tools import ToolRegistry, ToolResult, execute_code_in_sandbox
from .session import SessionStore, Session
from .stream import StreamEvent, EventType, EventCollector
from .verified_analysis import (
    VerifiedAnalyzer,
    Issue,
    AnalysisResult,
    Severity,
    Category,
    VerificationStatus,
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
]
