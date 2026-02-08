"""
Evolution Advisor Module

Uses Gemini to analyze codebases and provide strategic evolution recommendations:
- Technical debt identification
- Architecture improvements
- Feature roadmap suggestions
- Standards alignment

This is the "growth advisor" component - powered by Gemini.
"""

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any

import structlog

from .stream import StreamEvent, EventType

logger = structlog.get_logger()


class EvolutionCategory(str, Enum):
    """Categories of evolution recommendations."""
    TECHNICAL_DEBT = "technical_debt"
    ARCHITECTURE = "architecture"
    FEATURE = "feature"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEVOPS = "devops"


class Priority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Effort(str, Enum):
    """Effort estimates for recommendations."""
    TRIVIAL = "trivial"  # < 1 hour
    SMALL = "small"      # 1-4 hours
    MEDIUM = "medium"    # 1-3 days
    LARGE = "large"      # 1-2 weeks
    EPIC = "epic"        # > 2 weeks


@dataclass
class EvolutionRecommendation:
    """A single evolution recommendation."""
    id: str
    title: str
    description: str
    category: EvolutionCategory
    priority: Priority
    effort: Effort
    rationale: str
    impact: str
    implementation_steps: list[str]
    affected_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "effort": self.effort.value,
            "rationale": self.rationale,
            "impact": self.impact,
            "implementation_steps": self.implementation_steps,
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
        }


@dataclass
class EvolutionReport:
    """Complete evolution analysis report."""
    repo_url: str
    summary: str
    recommendations: list[EvolutionRecommendation]
    tech_stack_analysis: dict
    health_score: float  # 0-100
    maturity_level: str  # startup, growing, mature, legacy
    quick_wins: list[EvolutionRecommendation]
    strategic_initiatives: list[EvolutionRecommendation]
    analysis_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "summary": self.summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "tech_stack_analysis": self.tech_stack_analysis,
            "health_score": self.health_score,
            "maturity_level": self.maturity_level,
            "quick_wins": [r.to_dict() for r in self.quick_wins],
            "strategic_initiatives": [r.to_dict() for r in self.strategic_initiatives],
            "analysis_time_seconds": self.analysis_time_seconds,
        }


# Evolution analysis prompt template
EVOLUTION_ANALYSIS_PROMPT = '''You are an expert software architect and technical advisor analyzing a codebase for evolution opportunities.

CODEBASE TO ANALYZE:
{codebase_content}

ANALYSIS FOCUS:
{focus}

Perform a comprehensive analysis and provide:

1. **Tech Stack Analysis**:
   - Primary languages and frameworks
   - Dependencies and their health (outdated? deprecated?)
   - Architectural patterns used

2. **Code Health Assessment** (score 0-100):
   - Code organization and structure
   - Testing coverage estimation
   - Documentation quality
   - Error handling patterns
   - Security practices

3. **Maturity Level** (startup/growing/mature/legacy):
   - Based on code patterns, testing, documentation

4. **Evolution Recommendations** (minimum 5, maximum 15):
   For each recommendation provide:
   - title: Clear, actionable title
   - description: Detailed description
   - category: technical_debt, architecture, feature, performance, security, documentation, testing, devops
   - priority: critical, high, medium, low
   - effort: trivial (<1h), small (1-4h), medium (1-3 days), large (1-2 weeks), epic (>2 weeks)
   - rationale: Why this matters
   - impact: What improves when implemented
   - implementation_steps: Ordered list of steps
   - affected_files: Files that would be modified
   - dependencies: Other recommendations this depends on

5. **Quick Wins**: Top 3 low-effort, high-impact recommendations
6. **Strategic Initiatives**: Top 3 high-effort transformational recommendations

Return your analysis as JSON:
{{
  "tech_stack": {{
    "languages": ["python", "typescript"],
    "frameworks": ["fastapi", "react"],
    "databases": ["postgresql"],
    "infrastructure": ["docker", "kubernetes"],
    "outdated_deps": ["package@version - reason"],
    "patterns": ["microservices", "event-driven"]
  }},
  "health_score": 75,
  "maturity_level": "growing",
  "recommendations": [
    {{
      "id": "rec-001",
      "title": "Add input validation",
      "description": "...",
      "category": "security",
      "priority": "high",
      "effort": "medium",
      "rationale": "...",
      "impact": "...",
      "implementation_steps": ["Step 1", "Step 2"],
      "affected_files": ["src/api/users.py"],
      "dependencies": []
    }}
  ],
  "quick_wins": ["rec-001", "rec-003", "rec-005"],
  "strategic_initiatives": ["rec-002", "rec-007", "rec-010"],
  "summary": "Executive summary of findings and recommendations"
}}
'''

# Focused prompts for specific analysis areas
FOCUS_PROMPTS = {
    "full": "Analyze all aspects: architecture, security, performance, testing, documentation, and technical debt.",
    "architecture": "Focus on architectural patterns, modularity, separation of concerns, and scalability.",
    "security": "Focus on security practices, input validation, authentication, authorization, and data protection.",
    "performance": "Focus on performance bottlenecks, caching opportunities, database optimization, and resource usage.",
    "testing": "Focus on test coverage, test quality, testing patterns, and testability of the code.",
    "debt": "Focus on technical debt, code smells, deprecated patterns, and refactoring opportunities.",
    "devops": "Focus on CI/CD, infrastructure as code, monitoring, logging, and deployment practices.",
}


def parse_evolution_response(response_text: str) -> dict:
    """Parse the JSON response from Gemini."""
    # Try to extract JSON from markdown
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.error("evolution_json_parse_error", error=str(e))
            return {}
    return {}


class EvolutionAdvisor:
    """
    Gemini-powered codebase evolution advisor.
    
    Analyzes codebases and provides:
    - Technical debt assessment
    - Architecture recommendations
    - Feature roadmap suggestions
    - Quick wins for immediate improvement
    """
    
    def __init__(
        self,
        client,  # Gemini client
        model: str = "gemini-2.0-flash",
    ):
        self.client = client
        self.model = model
    
    async def analyze(
        self,
        repo_content: str,
        repo_url: str,
        focus: str = "full",
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> EvolutionReport:
        """
        Analyze codebase for evolution opportunities.
        
        Args:
            repo_content: Full repository content
            repo_url: Repository URL
            focus: Analysis focus (full, architecture, security, performance, testing, debt, devops)
            on_event: Callback for streaming events
        """
        import time
        import uuid
        
        start_time = time.time()
        
        def emit(event_type: EventType, data: dict):
            if on_event:
                on_event(StreamEvent(type=event_type, data=data))
        
        emit(EventType.THINKING, {
            "phase": "evolution_analysis",
            "message": f"Analyzing codebase for evolution opportunities (focus: {focus})..."
        })
        
        # Get focus-specific prompt
        focus_instruction = FOCUS_PROMPTS.get(focus, FOCUS_PROMPTS["full"])
        
        # Build analysis prompt
        prompt = EVOLUTION_ANALYSIS_PROMPT.format(
            codebase_content=repo_content[:400000],  # ~100K tokens
            focus=focus_instruction,
        )
        
        # Call Gemini for analysis
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            
            raw_response = response.text
            emit(EventType.TOKEN, {"content": raw_response[:500] + "...", "phase": "evolution"})
            
            # Parse response
            data = parse_evolution_response(raw_response)
            
            if not data:
                # Fallback if parsing fails
                return EvolutionReport(
                    repo_url=repo_url,
                    summary="Analysis completed but response parsing failed. Raw response available in logs.",
                    recommendations=[],
                    tech_stack_analysis={},
                    health_score=0.0,
                    maturity_level="unknown",
                    quick_wins=[],
                    strategic_initiatives=[],
                    analysis_time_seconds=time.time() - start_time,
                )
            
            # Build recommendations
            recommendations = []
            for rec_data in data.get("recommendations", []):
                try:
                    rec = EvolutionRecommendation(
                        id=rec_data.get("id", str(uuid.uuid4())[:8]),
                        title=rec_data.get("title", "Unknown"),
                        description=rec_data.get("description", ""),
                        category=EvolutionCategory(rec_data.get("category", "technical_debt")),
                        priority=Priority(rec_data.get("priority", "medium")),
                        effort=Effort(rec_data.get("effort", "medium")),
                        rationale=rec_data.get("rationale", ""),
                        impact=rec_data.get("impact", ""),
                        implementation_steps=rec_data.get("implementation_steps", []),
                        affected_files=rec_data.get("affected_files", []),
                        dependencies=rec_data.get("dependencies", []),
                    )
                    recommendations.append(rec)
                    
                    emit(EventType.TOOL_RESULT, {
                        "name": "recommendation_found",
                        "recommendation": rec.to_dict()
                    })
                except (ValueError, KeyError) as e:
                    logger.warning("recommendation_parse_error", error=str(e))
                    continue
            
            # Build quick wins and strategic lists
            quick_win_ids = set(data.get("quick_wins", []))
            strategic_ids = set(data.get("strategic_initiatives", []))
            
            quick_wins = [r for r in recommendations if r.id in quick_win_ids]
            strategic = [r for r in recommendations if r.id in strategic_ids]
            
            # If IDs didn't match, use heuristics
            if not quick_wins:
                quick_wins = sorted(
                    [r for r in recommendations if r.effort in (Effort.TRIVIAL, Effort.SMALL)],
                    key=lambda r: (
                        {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}[r.priority]
                    )
                )[:3]
            
            if not strategic:
                strategic = sorted(
                    [r for r in recommendations if r.effort in (Effort.LARGE, Effort.EPIC)],
                    key=lambda r: (
                        {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}[r.priority]
                    )
                )[:3]
            
            report = EvolutionReport(
                repo_url=repo_url,
                summary=data.get("summary", "Analysis complete."),
                recommendations=recommendations,
                tech_stack_analysis=data.get("tech_stack", {}),
                health_score=float(data.get("health_score", 50)),
                maturity_level=data.get("maturity_level", "unknown"),
                quick_wins=quick_wins,
                strategic_initiatives=strategic,
                analysis_time_seconds=time.time() - start_time,
            )
            
            emit(EventType.THINKING, {
                "phase": "evolution_complete",
                "message": f"Found {len(recommendations)} evolution recommendations",
                "health_score": report.health_score,
            })
            
            return report
            
        except Exception as e:
            logger.error("evolution_analysis_error", error=str(e))
            raise
    
    async def generate_roadmap(
        self,
        recommendations: list[EvolutionRecommendation],
        constraints: dict = None,
    ) -> dict:
        """
        Generate a prioritized implementation roadmap.
        
        Args:
            recommendations: List of recommendations to schedule
            constraints: Optional constraints (team_size, timeline, etc.)
        """
        # Group by priority and effort
        phases = {
            "immediate": [],  # Critical + trivial/small effort
            "short_term": [],  # High + small/medium effort
            "medium_term": [],  # Medium priority or medium effort
            "long_term": [],  # Large/epic effort
        }
        
        for rec in recommendations:
            if rec.priority == Priority.CRITICAL and rec.effort in (Effort.TRIVIAL, Effort.SMALL):
                phases["immediate"].append(rec)
            elif rec.priority in (Priority.CRITICAL, Priority.HIGH) and rec.effort in (Effort.SMALL, Effort.MEDIUM):
                phases["short_term"].append(rec)
            elif rec.effort in (Effort.LARGE, Effort.EPIC):
                phases["long_term"].append(rec)
            else:
                phases["medium_term"].append(rec)
        
        return {
            "phases": {
                phase: [r.to_dict() for r in recs]
                for phase, recs in phases.items()
            },
            "total_items": len(recommendations),
            "estimated_total_effort": self._estimate_total_effort(recommendations),
        }
    
    def _estimate_total_effort(self, recommendations: list[EvolutionRecommendation]) -> str:
        """Estimate total effort in human-readable format."""
        effort_hours = {
            Effort.TRIVIAL: 0.5,
            Effort.SMALL: 2.5,
            Effort.MEDIUM: 16,  # 2 days
            Effort.LARGE: 60,  # 1.5 weeks
            Effort.EPIC: 120,  # 3 weeks
        }
        
        total_hours = sum(effort_hours.get(r.effort, 16) for r in recommendations)
        
        if total_hours < 8:
            return f"{total_hours:.0f} hours"
        elif total_hours < 40:
            return f"{total_hours/8:.1f} days"
        else:
            return f"{total_hours/40:.1f} weeks"
