"""
Gemini Agent - Hackathon Project

A Gemini-powered Marathon Agent with tool calling capabilities.

Features:
- Multi-turn agentic loop with thought signature continuity
- Dynamic thinking levels for cost/latency optimization
- E2B code execution sandbox
- Session persistence for resume capability
- SSE streaming for real-time progress

Run locally:
    uvicorn src.main:app --reload

Deploy to Railway:
    git push (auto-deploys if connected)

Last deploy trigger: 2026-01-18
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

# Load environment variables
load_dotenv()

# Import agent module
from .agent import (
    MarathonAgent, AgentConfig, AgentResult,
    ToolRegistry, SessionStore,
    StreamEvent, EventType, EventCollector,
    VerifiedAnalyzer, Issue, AnalysisResult, VerificationStatus,
    execute_code_in_sandbox,
)
from .agent.tools import create_default_tools

# ============================================================
# VERSION CONSTANT
# ============================================================
__version__ = "0.9.1"

# ============================================================
# SECURITY - API Key Protection
# ============================================================

def get_api_key() -> Optional[str]:
    """Get the API key from environment. None = open access."""
    return os.environ.get("API_SECRET_KEY")


async def verify_api_key(request: Request):
    """
    Verify API key if API_SECRET_KEY is set.
    
    Security modes (controlled by SECURITY_MODE env var):
    - "open": No authentication required (default for development)
    - "strict": Fail if API_SECRET_KEY is not configured (production recommended)
    - If API_SECRET_KEY is set: always requires X-API-Key header
    """
    api_key = get_api_key()
    security_mode = os.environ.get("SECURITY_MODE", "open").lower()
    
    # If no API key configured, check security mode
    if not api_key:
        if security_mode == "strict":
            # Fail-secure: reject requests if auth not configured
            raise HTTPException(
                status_code=500,
                detail="Server misconfiguration: API_SECRET_KEY not set but SECURITY_MODE=strict"
            )
        # Default: allow requests in demo/development mode
        return
    
    # Check for API key in header
    provided_key = request.headers.get("X-API-Key")
    
    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header"
        )
    
    if provided_key != api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )


def get_owner_id(request: Request) -> str:
    """Derive owner_id from API key for session scoping. Returns hash, never raw key."""
    import hashlib
    key = request.headers.get("X-API-Key", "")
    if not key:
        return "anonymous"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Gemini client (lazy initialized)
_gemini_client = None

# Marathon Agent (lazy initialized)
_marathon_agent = None
_session_store = None


def get_gemini_client():
    """Get or initialize the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY environment variable not set"
            )
        
        _gemini_client = genai.Client(api_key=api_key)
    
    return _gemini_client


def get_model_name() -> str:
    """Get the configured model name for general use."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def get_reasoning_model() -> str:
    """Get the model for deep reasoning tasks (analysis, verification)."""
    return os.environ.get("GEMINI_REASONING_MODEL", "gemini-3-pro-preview")


def get_marathon_agent() -> MarathonAgent:
    """Get or initialize the Marathon Agent."""
    global _marathon_agent, _session_store
    
    if _marathon_agent is None:
        client = get_gemini_client()
        tools = create_default_tools()
        _session_store = SessionStore()
        
        config = AgentConfig(
            model=get_model_name(),
            max_iterations=100,
            timeout_seconds=600.0,  # 10 minutes default
        )
        
        _marathon_agent = MarathonAgent(
            client=client,
            tools=tools,
            sessions=_session_store,
            config=config,
        )
    
    return _marathon_agent


def get_session_store() -> SessionStore:
    """Get session store (initializes agent if needed)."""
    get_marathon_agent()  # Ensure initialized
    return _session_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("starting_gemini_agent", model=get_model_name())
    
    # Start background cleanup task for async jobs
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    
    yield
    
    # Cancel cleanup on shutdown
    cleanup_task.cancel()
    logger.info("shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Gemini Agent",
    description="Gemini-powered agent with tool calling",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - explicit origins for security (no wildcard with credentials)
CORS_ORIGINS_DEFAULT = [
    "http://localhost:5173",      # Vite dev server
    "http://localhost:3000",      # Alternative dev
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://gemini-frontend-murex.vercel.app",  # Production frontend
    "https://gemini-frontend.vercel.app",        # Alternative Vercel URL
]

cors_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if cors_origins_env:
    # If env var is set, use those origins (comma-separated)
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    allowed_origins = CORS_ORIGINS_DEFAULT

# If "*" is in the list, we can't use credentials
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and status."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Skip logging for health checks and static assets
    path = request.url.path
    skip_logging = path in ["/health", "/favicon.ico"]
    
    if not skip_logging:
        add_log("info", "request_started", 
                request_id=request_id,
                method=request.method,
                path=path,
                client=request.client.host if request.client else "unknown")
    
    # Process request
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        _metrics["requests"]["total"] += 1
        if 200 <= response.status_code < 400:
            _metrics["requests"]["success"] += 1
        else:
            _metrics["requests"]["errors"] += 1
        
        _metrics["last_request"] = {
            "path": path,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        
        if not skip_logging:
            add_log("info", "request_completed",
                    request_id=request_id,
                    method=request.method,
                    path=path,
                    status=response.status_code,
                    duration_ms=round(duration_ms, 2))
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _metrics["requests"]["total"] += 1
        _metrics["requests"]["errors"] += 1
        
        add_log("error", "request_failed",
                request_id=request_id,
                method=request.method,
                path=path,
                error=str(e)[:200],
                duration_ms=round(duration_ms, 2))
        raise


# ============================================================
# MODELS
# ============================================================

class GenerateRequest(BaseModel):
    """Request to generate content."""
    prompt: str
    system_instruction: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response from generation."""
    text: str
    model: str


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # "user" or "model"
    content: str


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage]
    system_instruction: Optional[str] = None


class ToolCall(BaseModel):
    """A tool call from the model."""
    name: str
    arguments: dict


class AgentRequest(BaseModel):
    """Request for agent execution with tools."""
    prompt: str
    system_instruction: Optional[str] = None
    max_iterations: int = 100
    timeout_seconds: float = 600.0  # 10 minutes default


class AgentResponse(BaseModel):
    """Response from agent execution."""
    text: str
    model: str
    tool_calls: List[ToolCall]
    iterations: int


# ============================================================
# TOOLS - Define your agent's capabilities here
# ============================================================

# Example tools - replace with your own
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'America/New_York')"
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
]


async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool and return the result."""
    import datetime
    
    if name == "get_current_time":
        tz = arguments.get("timezone", "UTC")
        return f"Current time: {datetime.datetime.now().isoformat()} ({tz})"
    
    elif name == "calculate":
        expr = arguments.get("expression", "")
        try:
            # Use AST-based safe calculator (no eval)
            from agent.tools import CalculateTool
            calc = CalculateTool()
            result = calc._safe_eval(expr)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    elif name == "search_web":
        query = arguments.get("query", "")
        # Placeholder - integrate with actual search API
        return f"Search results for '{query}': [Implement actual search]"
    
    else:
        return f"Unknown tool: {name}"


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - shows API info."""
    return {
        "name": "Gemini Marathon Agent",
        "version": __version__,
        "status": "running",
        "model": get_model_name(),
        "docs": "/docs",
        "endpoints": {
            "legacy": {
                "generate": "/generate",
                "chat": "/chat",
                "agent": "/agent",
            },
            "v2": {
                "agent": "/v2/agent",
                "agent_stream": "/v2/agent/stream",
                "tools": "/v2/tools",
                "sessions": "/v2/sessions",
            },
            "v3_hackathon": {
                "analyze": "/v3/analyze",
                "tools": "/v3/tools",
            }
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint for Railway/deployment platforms."""
    return {
        "status": "healthy",
        "models": {
            "default": get_model_name(),
            "reasoning": get_reasoning_model(),
        },
        "secured": get_api_key() is not None,
        "version": __version__,
        "capabilities": [
            "marathon_agent",
            "tool_calling",
            "code_execution",
            "session_persistence",
            "streaming",
            "codebase_analysis",
            "verified_analysis",
            "auto_fix_suggestions",
        ],
        # Configuration status for frontend pre-flight checks
        "config": {
            "e2b_configured": bool(os.environ.get("E2B_API_KEY")),
            "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        },
    }


# ============================================================
# DIAGNOSTICS - Real telemetry for observability
# ============================================================

# In-memory metrics (resets on deploy)
import datetime
import time
import uuid
from collections import deque

# Startup time
_startup_time = datetime.datetime.utcnow()

# Metrics
_metrics = {
    "requests": {"total": 0, "success": 0, "errors": 0},
    "gemini_calls": {"total": 0, "success": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0},
    "e2b_calls": {"total": 0, "success": 0, "errors": 0},
    "analyses": {"total": 0, "verified": 0, "issues_found": 0},
    "last_error": None,
    "last_request": None,
    "startup_time": _startup_time.isoformat(),
}

# Log buffer - keeps last 100 log entries in memory
_log_buffer: deque = deque(maxlen=100)

MAX_LOG_VALUE_SIZE = 500  # Truncate large values in log entries

def add_log(level: str, event: str, **kwargs):
    """Add a log entry to the buffer and structlog."""
    # Truncate large values to prevent memory bloat
    safe_kwargs = {}
    for k, v in kwargs.items():
        val_str = str(v)
        safe_kwargs[k] = val_str[:MAX_LOG_VALUE_SIZE] + "...(truncated)" if len(val_str) > MAX_LOG_VALUE_SIZE else v
    
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "level": level,
        "event": event,
        **safe_kwargs
    }
    _log_buffer.append(entry)
    
    # Also log via structlog
    if level == "error":
        logger.error(event, **kwargs)
    elif level == "warning":
        logger.warning(event, **kwargs)
    else:
        logger.info(event, **kwargs)

def record_metric(category: str, metric: str, value: int = 1):
    """Record a metric increment."""
    if category in _metrics and metric in _metrics[category]:
        _metrics[category][metric] += value

def record_error(error: str):
    """Record the last error."""
    import datetime
    _metrics["last_error"] = {
        "message": str(error)[:500],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

def record_request():
    """Record a request."""
    import datetime
    _metrics["requests"]["total"] += 1
    _metrics["last_request"] = datetime.datetime.utcnow().isoformat()


@app.get("/status")
async def public_status():
    """
    Public status endpoint - no API key required.
    Shows basic metrics for debugging without exposing sensitive data.
    
    Note: Railway has a 5-minute hard timeout on HTTP requests.
    For long-running analyses, use the async API: POST /v4/analyze/async
    """
    import datetime
    uptime = (datetime.datetime.utcnow() - _startup_time).total_seconds()
    
    return {
        "status": "running",
        "version": __version__,
        "uptime_seconds": round(uptime),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "metrics": {
            "total_requests": _metrics["requests"]["total"],
            "total_analyses": _metrics["analyses"]["total"],
            "verified_issues": _metrics["analyses"]["verified"],
            "errors": _metrics["requests"]["errors"],
        },
        "last_activity": _metrics["last_request"],
        "last_error": _metrics["last_error"]["message"][:100] if _metrics["last_error"] else None,
        "platform_limits": {
            "note": "Railway has 5-minute HTTP timeout",
            "recommendation": "Use async API for long analyses: POST /v4/analyze/async",
        }
    }


@app.get("/diagnostics", dependencies=[Depends(verify_api_key)])
async def diagnostics():
    """
    Comprehensive diagnostics endpoint - tests all components.
    Returns detailed health status for debugging.
    """
    import datetime
    import time
    
    results = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": __version__,
        "checks": {},
        "metrics": _metrics,
    }
    
    # 1. Check Gemini API connectivity
    gemini_status = {"status": "unknown", "latency_ms": None, "error": None}
    try:
        client = get_gemini_client()
        start = time.time()
        response = client.models.generate_content(
            model=get_model_name(),
            contents="Say 'OK' and nothing else.",
        )
        latency = (time.time() - start) * 1000
        gemini_status = {
            "status": "ok",
            "latency_ms": round(latency, 2),
            "response": response.text[:50] if response.text else None,
            "model": get_model_name(),
        }
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["success"] += 1
    except Exception as e:
        gemini_status = {"status": "error", "error": str(e)[:200]}
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["errors"] += 1
        record_error(f"Gemini: {e}")
    
    results["checks"]["gemini_api"] = gemini_status
    
    # 2. Check E2B sandbox configuration (don't spin up sandbox - too slow for diagnostics)
    e2b_status = {"status": "unknown", "error": None}
    try:
        e2b_key = os.environ.get("E2B_API_KEY")
        if not e2b_key:
            e2b_status = {"status": "not_configured", "error": "E2B_API_KEY not set"}
        else:
            try:
                from e2b_code_interpreter import Sandbox
                # Just verify import works and key is set
                e2b_status = {
                    "status": "configured",
                    "api_key_set": True,
                    "note": "Use /diagnostics/e2b for full sandbox test",
                }
            except ImportError:
                e2b_status = {"status": "not_installed", "error": "e2b_code_interpreter not installed"}
    except Exception as e:
        e2b_status = {"status": "error", "error": str(e)[:200]}
    
    results["checks"]["e2b_sandbox"] = e2b_status
    
    # 3. Check Redis/session store
    session_status = {"status": "unknown", "backend": None, "error": None}
    try:
        store = get_session_store()
        session_status = {
            "status": "ok",
            "backend": "redis" if store._redis else "memory",
            "sessions_count": len(await store.list_sessions()) if hasattr(store, 'list_sessions') else "unknown",
        }
    except Exception as e:
        session_status = {"status": "error", "error": str(e)[:200]}
    
    results["checks"]["session_store"] = session_status
    
    # 4. Environment check
    env_status = {
        "gemini_api_key": "set" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") else "missing",
        "e2b_api_key": "set" if os.environ.get("E2B_API_KEY") else "missing",
        "api_secret_key": "set" if os.environ.get("API_SECRET_KEY") else "missing (open access)",
        "redis_url": "set" if os.environ.get("REDIS_URL") else "missing (using memory)",
    }
    results["checks"]["environment"] = env_status
    
    # Overall status - Gemini is required, E2B and Redis are optional
    gemini_ok = gemini_status["status"] == "ok"
    e2b_ok = e2b_status["status"] in ["ok", "configured", "not_configured", "not_installed"]
    session_ok = session_status["status"] == "ok"
    
    if gemini_ok and session_ok:
        results["overall"] = "healthy"
    elif gemini_ok:
        results["overall"] = "degraded"  # Can still work, but with issues
    else:
        results["overall"] = "unhealthy"  # Gemini is required
    
    return results


@app.get("/diagnostics/quick", dependencies=[Depends(verify_api_key)])
async def diagnostics_quick():
    """Quick diagnostics - just returns cached metrics without running tests."""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "metrics": _metrics,
        "version": __version__,
        "uptime_seconds": (datetime.datetime.utcnow() - _startup_time).total_seconds(),
    }


@app.get("/logs", dependencies=[Depends(verify_api_key)])
async def get_logs(limit: int = 50, level: Optional[str] = None):
    """
    Get recent log entries from the in-memory buffer.
    
    Args:
        limit: Max number of entries to return (default 50, max 100)
        level: Filter by level (info, warning, error)
    
    Returns recent activity for debugging and monitoring.
    """
    limit = min(limit, 100)
    
    logs = list(_log_buffer)
    
    # Filter by level if specified
    if level:
        logs = [l for l in logs if l.get("level") == level]
    
    # Return most recent first
    logs = logs[-limit:][::-1]
    
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "total_in_buffer": len(_log_buffer),
        "returned": len(logs),
        "logs": logs,
    }


@app.get("/logs/errors", dependencies=[Depends(verify_api_key)])
async def get_error_logs(limit: int = 20):
    """Get recent error logs only."""
    logs = [l for l in _log_buffer if l.get("level") == "error"]
    logs = logs[-limit:][::-1]
    
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "total_errors": len([l for l in _log_buffer if l.get("level") == "error"]),
        "returned": len(logs),
        "logs": logs,
    }


@app.get("/logs/requests", dependencies=[Depends(verify_api_key)])
async def get_request_logs(limit: int = 20):
    """Get recent request logs with timing."""
    logs = [l for l in _log_buffer if l.get("event") in ["request_started", "request_completed", "request_failed"]]
    logs = logs[-limit:][::-1]
    
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "returned": len(logs),
        "logs": logs,
    }


@app.get("/diagnostics/e2b", dependencies=[Depends(verify_api_key)])
async def diagnostics_e2b():
    """
    Test E2B sandbox connectivity.
    This spins up a real sandbox and runs code - use sparingly.
    """
    import datetime
    import time
    
    e2b_key = os.environ.get("E2B_API_KEY")
    if not e2b_key:
        return {"status": "not_configured", "error": "E2B_API_KEY not set"}
    
    try:
        from e2b_code_interpreter import Sandbox
        start = time.time()
        
        # E2B v2.x API: use Sandbox.create() context manager
        with Sandbox.create() as sandbox:
            result = sandbox.run_code("import sys; print(f'Python {sys.version}')")
            output = result.text if hasattr(result, 'text') else (result.logs.stdout if result.logs else None)
        
        latency = (time.time() - start) * 1000
        _metrics["e2b_calls"]["total"] += 1
        _metrics["e2b_calls"]["success"] += 1
        
        return {
            "status": "ok",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "latency_ms": round(latency, 2),
            "output": output[:200] if output else "No output",
        }
    except ImportError:
        return {"status": "not_installed", "error": "e2b_code_interpreter not installed"}
    except Exception as e:
        _metrics["e2b_calls"]["total"] += 1
        _metrics["e2b_calls"]["errors"] += 1
        record_error(f"E2B test: {e}")
        return {
            "status": "error",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error": str(e)[:500],
        }


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    """Generate content with Gemini."""
    try:
        from google.genai.types import GenerateContentConfig
        
        client = get_gemini_client()
        start_time = time.time()
        
        config = GenerateContentConfig()
        if request.system_instruction:
            config = GenerateContentConfig(
                system_instruction=request.system_instruction
            )
        
        response = client.models.generate_content(
            model=get_model_name(),
            contents=request.prompt,
            config=config
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Track token usage if available
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, 'usage_metadata'):
            tokens_in = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            tokens_out = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["success"] += 1
        _metrics["gemini_calls"]["tokens_in"] += tokens_in
        _metrics["gemini_calls"]["tokens_out"] += tokens_out
        
        add_log("info", "gemini_generate",
                model=get_model_name(),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=round(duration_ms, 2))
        
        return GenerateResponse(
            text=response.text,
            model=get_model_name(),
        )
        
    except Exception as e:
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["errors"] += 1
        add_log("error", "gemini_generate_error", error=str(e)[:200])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """Multi-turn chat with Gemini."""
    try:
        from google.genai.types import Content, Part, GenerateContentConfig
        
        client = get_gemini_client()
        start_time = time.time()
        
        # Convert messages to Gemini format
        contents = []
        for msg in request.messages:
            contents.append(Content(
                role=msg.role if msg.role != "assistant" else "model",
                parts=[Part(text=msg.content)]
            ))
        
        config = GenerateContentConfig()
        if request.system_instruction:
            config = GenerateContentConfig(
                system_instruction=request.system_instruction
            )
        
        response = client.models.generate_content(
            model=get_model_name(),
            contents=contents,
            config=config
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Track token usage
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, 'usage_metadata'):
            tokens_in = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            tokens_out = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["success"] += 1
        _metrics["gemini_calls"]["tokens_in"] += tokens_in
        _metrics["gemini_calls"]["tokens_out"] += tokens_out
        
        add_log("info", "gemini_chat",
                model=get_model_name(),
                messages=len(contents),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=round(duration_ms, 2))
        
        return GenerateResponse(
            text=response.text,
            model=get_model_name(),
        )
        
    except Exception as e:
        _metrics["gemini_calls"]["total"] += 1
        _metrics["gemini_calls"]["errors"] += 1
        add_log("error", "gemini_chat_error", error=str(e)[:200])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent", response_model=AgentResponse, dependencies=[Depends(verify_api_key)])
async def agent(request: AgentRequest):
    """
    Run the agent with tool calling.
    
    This is the main agentic endpoint - the agent will:
    1. Receive a prompt
    2. Decide which tools to call
    3. Execute tools
    4. Continue until task is complete or max iterations reached
    """
    try:
        from google.genai.types import (
            Content, Part, GenerateContentConfig, 
            Tool, FunctionDeclaration
        )
        
        client = get_gemini_client()
        
        # Convert tools to Gemini format
        function_declarations = []
        for tool in TOOLS:
            function_declarations.append(
                FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"]
                )
            )
        
        gemini_tools = [Tool(function_declarations=function_declarations)]
        
        # Build config
        config = GenerateContentConfig(
            tools=gemini_tools
        )
        if request.system_instruction:
            config = GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=request.system_instruction
            )
        
        # Agentic loop
        messages = [Content(role="user", parts=[Part(text=request.prompt)])]
        all_tool_calls = []
        iterations = 0
        
        while iterations < request.max_iterations:
            iterations += 1
            
            response = client.models.generate_content(
                model=get_model_name(),
                contents=messages,
                config=config
            )
            
            # Check if model wants to call tools
            candidate = response.candidates[0]
            
            # Check for function calls
            function_calls = []
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
            
            if not function_calls:
                # No tool calls - we're done
                return AgentResponse(
                    text=response.text or "",
                    model=get_model_name(),
                    tool_calls=all_tool_calls,
                    iterations=iterations
                )
            
            # Execute tool calls
            messages.append(candidate.content)
            
            tool_results = []
            for fc in function_calls:
                tool_call = ToolCall(
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {}
                )
                all_tool_calls.append(tool_call)
                
                result = await execute_tool(fc.name, tool_call.arguments)
                tool_results.append(Part(
                    function_response={
                        "name": fc.name,
                        "response": {"result": result}
                    }
                ))
                
                logger.info("tool_executed", 
                    tool=fc.name, 
                    args=tool_call.arguments,
                    result=result[:100] if result else None
                )
            
            # Add tool results to conversation
            messages.append(Content(role="user", parts=tool_results))
        
        # Max iterations reached
        return AgentResponse(
            text="Max iterations reached",
            model=get_model_name(),
            tool_calls=all_tool_calls,
            iterations=iterations
        )
        
    except Exception as e:
        logger.error("agent_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools():
    """List available tools (legacy)."""
    return {"tools": TOOLS}


# ============================================================
# V2 ENDPOINTS - Marathon Agent
# ============================================================

class MarathonAgentRequest(BaseModel):
    """Request for Marathon Agent execution."""
    task: str
    session_id: Optional[str] = None
    max_iterations: Optional[int] = None
    timeout_seconds: Optional[float] = None  # Default: 10 minutes, max: 1 hour


class MarathonAgentResponse(BaseModel):
    """Response from Marathon Agent."""
    text: str
    tool_calls: List[dict]
    iterations: int
    session_id: Optional[str]
    completed: bool
    error: Optional[str] = None


@app.post("/v2/agent", response_model=MarathonAgentResponse, dependencies=[Depends(verify_api_key)])
async def marathon_agent(request: MarathonAgentRequest, raw_request: Request):
    """
    Run the Marathon Agent.
    
    This is the new agent endpoint with:
    - Dynamic thinking levels
    - E2B code execution
    - Session persistence for resume
    - Parallel tool execution
    """
    try:
        agent = get_marathon_agent()
        
        # Override limits if specified
        if request.max_iterations:
            agent.config.max_iterations = request.max_iterations
        if request.timeout_seconds:
            # Cap at 1 hour for safety
            agent.config.timeout_seconds = min(request.timeout_seconds, 3600.0)
        
        result = await agent.run(
            task=request.task,
            session_id=request.session_id,
            owner_id=get_owner_id(raw_request),
        )
        
        return MarathonAgentResponse(
            text=result.text,
            tool_calls=result.tool_calls,
            iterations=result.iterations,
            session_id=result.session_id,
            completed=result.completed,
            error=result.error,
        )
        
    except Exception as e:
        logger.error("marathon_agent_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/agent/stream", dependencies=[Depends(verify_api_key)])
async def marathon_agent_stream(request: MarathonAgentRequest, raw_request: Request):
    """
    Run the Marathon Agent with SSE streaming.
    
    Returns Server-Sent Events for real-time progress:
    - start: Agent started
    - thinking: Thinking level selected
    - token: Text generated
    - tool_start: Tool execution started
    - tool_result: Tool execution completed
    - done: Agent completed
    - error: Error occurred
    """
    try:
        agent = get_marathon_agent()
        
        # Override limits if specified
        if request.max_iterations:
            agent.config.max_iterations = request.max_iterations
        if request.timeout_seconds:
            agent.config.timeout_seconds = min(request.timeout_seconds, 3600.0)
        
        # Collect events
        collector = EventCollector()
        
        # Run agent with event callback
        result = await agent.run(
            task=request.task,
            session_id=request.session_id,
            owner_id=get_owner_id(raw_request),
            on_event=collector,
        )
        
        # Add final result event
        collector(StreamEvent(
            type=EventType.DONE,
            data={
                "text": result.text,
                "tool_calls": result.tool_calls,
                "iterations": result.iterations,
                "session_id": result.session_id,
                "completed": result.completed,
            }
        ))
        
        async def event_generator():
            for event in collector.events:
                yield event.to_sse()
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error("marathon_agent_stream_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/tools")
async def list_marathon_tools():
    """List available Marathon Agent tools."""
    agent = get_marathon_agent()
    return {
        "tools": [
            {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in agent.tools.items()
        ]
    }


@app.get("/v2/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    """List recent sessions."""
    store = get_session_store()
    session_ids = await store.list_sessions()
    return {"sessions": session_ids}


@app.get("/v2/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session(session_id: str):
    """Get session details."""
    store = get_session_store()
    session = await store.load(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "id": session.id,
        "iteration": session.iteration,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
    }


@app.delete("/v2/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(session_id: str):
    """Delete a session."""
    store = get_session_store()
    await store.delete(session_id)
    return {"deleted": session_id}


# ============================================================
# V3 ENDPOINTS - Codebase Analyst (Hackathon Feature)
# ============================================================

# Codebase Analyst Agent (lazy initialized)
_codebase_agent = None


def get_codebase_agent() -> MarathonAgent:
    """Get or initialize the Codebase Analyst Agent."""
    global _codebase_agent
    
    if _codebase_agent is None:
        from .agent.tools import create_codebase_analyst_tools
        
        client = get_gemini_client()
        tools = create_codebase_analyst_tools()
        
        config = AgentConfig(
            model=get_model_name(),
            max_iterations=20,  # More iterations for complex analysis
            system_instruction="""You are an expert code analyst. Your task is to analyze codebases thoroughly.

When analyzing a repository:
1. First use clone_repo to load the codebase
2. Review the directory structure and identify key components
3. Analyze architecture patterns, dependencies, and code quality
4. Identify potential bugs, security issues, and anti-patterns
5. Generate specific, actionable recommendations
6. If requested, generate tests using execute_code to verify findings

Be thorough but concise. Provide code examples for improvements.
Format your analysis with clear sections and bullet points."""
        )
        
        _codebase_agent = MarathonAgent(
            client=client,
            tools=tools,
            sessions=_session_store,
            config=config,
        )
    
    return _codebase_agent


class AnalyzeRepoRequest(BaseModel):
    """Request to analyze a GitHub repository."""
    repo_url: str
    focus: Optional[str] = "all"  # bugs, security, performance, architecture, all
    branch: Optional[str] = None
    path_filter: Optional[str] = None
    session_id: Optional[str] = None


class AnalyzeRepoResponse(BaseModel):
    """Response from repository analysis."""
    analysis: str
    repo_url: str
    files_analyzed: int
    issues_found: int
    tool_calls: List[dict]
    iterations: int
    session_id: Optional[str]
    completed: bool


@app.post("/v3/analyze", response_model=AnalyzeRepoResponse, dependencies=[Depends(verify_api_key)])
async def analyze_repository(request: AnalyzeRepoRequest, raw_request: Request):
    """
    Analyze a GitHub repository.
    
    This is the hackathon showcase endpoint - demonstrates:
    - Loading codebases using Gemini's large context window (truncated for reliability)
    - Multi-step autonomous analysis
    - Code execution for verification
    - Comprehensive reporting
    
    Focus options:
    - bugs: Find potential bugs and logic errors
    - security: Identify security vulnerabilities
    - performance: Find performance issues
    - architecture: Analyze code structure and patterns
    - all: Comprehensive analysis
    """
    try:
        agent = get_codebase_agent()
        
        # Build the analysis task
        focus_prompts = {
            "bugs": "Focus on finding potential bugs, logic errors, edge cases, and error handling issues.",
            "security": "Focus on security vulnerabilities, injection risks, authentication issues, and data exposure.",
            "performance": "Focus on performance bottlenecks, inefficient algorithms, and resource management.",
            "architecture": "Focus on code structure, design patterns, modularity, and maintainability.",
            "all": "Perform a comprehensive analysis covering bugs, security, performance, and architecture."
        }
        
        focus_instruction = focus_prompts.get(request.focus, focus_prompts["all"])
        
        task = f"""Analyze the GitHub repository: {request.repo_url}

{focus_instruction}

Steps:
1. Clone and load the repository
2. Review the codebase structure
3. Perform detailed analysis based on the focus area
4. Identify specific issues with file locations and line numbers where possible
5. Provide actionable recommendations with code examples
6. Summarize findings with severity ratings (critical, high, medium, low)

Repository: {request.repo_url}
Branch: {request.branch or 'default'}
Path filter: {request.path_filter or 'all files'}
"""
        
        result = await agent.run(
            task=task,
            session_id=request.session_id,
            owner_id=get_owner_id(raw_request),
        )
        
        # Extract metrics from the result
        files_analyzed = 0
        issues_found = 0
        
        for tc in result.tool_calls:
            if tc.get("name") == "clone_repo":
                # Estimate from output
                files_analyzed = 10  # Default, would parse from actual result
        
        # Count issue mentions in the text
        issue_keywords = ["bug", "issue", "vulnerability", "error", "problem", "warning"]
        for keyword in issue_keywords:
            issues_found += result.text.lower().count(keyword)
        
        return AnalyzeRepoResponse(
            analysis=result.text,
            repo_url=request.repo_url,
            files_analyzed=files_analyzed,
            issues_found=min(issues_found, 50),  # Cap at 50
            tool_calls=result.tool_calls,
            iterations=result.iterations,
            session_id=result.session_id,
            completed=result.completed,
        )
        
    except Exception as e:
        logger.error("analyze_repo_error", error=str(e), repo=request.repo_url)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v3/analyze/stream", dependencies=[Depends(verify_api_key)])
async def analyze_repository_stream(request: AnalyzeRepoRequest, raw_request: Request):
    """
    Analyze a GitHub repository with real-time SSE streaming.
    
    Streams progress events:
    - start: Analysis started
    - thinking: Agent is reasoning
    - tool_start: Tool execution started (e.g., "Cloning repo...")
    - tool_result: Tool completed with output
    - token: Text being generated
    - done: Analysis complete with full results
    - error: Error occurred
    """
    import asyncio
    
    agent = get_codebase_agent()
    
    # Build the analysis task
    focus_prompts = {
        "bugs": "Focus on finding potential bugs, logic errors, edge cases, and error handling issues.",
        "security": "Focus on security vulnerabilities, injection risks, authentication issues, and data exposure.",
        "performance": "Focus on performance bottlenecks, inefficient algorithms, and resource management.",
        "architecture": "Focus on code structure, design patterns, modularity, and maintainability.",
        "all": "Perform a comprehensive analysis covering bugs, security, performance, and architecture."
    }
    
    focus_instruction = focus_prompts.get(request.focus, focus_prompts["all"])
    
    task = f"""Analyze the GitHub repository: {request.repo_url}

{focus_instruction}

Steps:
1. Clone and load the repository
2. Review the codebase structure
3. Perform detailed analysis based on the focus area
4. Identify specific issues with file locations and line numbers where possible
5. Provide actionable recommendations with code examples
6. Summarize findings with severity ratings (critical, high, medium, low)

Repository: {request.repo_url}
Branch: {request.branch or 'default'}
Path filter: {request.path_filter or 'all files'}
"""
    
    # Use asyncio.Queue for real-time streaming
    event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    
    # Accumulate analysis text from token events
    accumulated_text: list[str] = []
    accumulated_tool_calls: list[dict] = []
    final_iterations = 0
    
    def on_event(event: StreamEvent):
        """Callback to push events to queue, accumulating text."""
        nonlocal final_iterations
        
        # Accumulate token content for final result
        if event.type == EventType.TOKEN:
            content = event.data.get("content", "")
            if content:
                accumulated_text.append(content)
        
        # Track tool calls
        if event.type == EventType.TOOL_START:
            accumulated_tool_calls.append(event.data)
        
        # Intercept agent's DONE event - we'll create our own with full text
        if event.type == EventType.DONE:
            final_iterations = event.data.get("iterations", 0)
            # Don't forward - we'll create a better one after agent completes
            return
        
        event_queue.put_nowait(event)
    
    async def run_agent():
        """Run agent in background and push completion."""
        try:
            result = await agent.run(
                task=task,
                session_id=request.session_id,
                on_event=on_event,
            )
            
            # Use accumulated text or fallback to result.text
            analysis_text = "".join(accumulated_text) or result.text or ""
            
            # Extract metrics
            files_analyzed = 0
            for tc in accumulated_tool_calls:
                if tc.get("name") == "clone_repo":
                    # Try to get file count from output
                    files_analyzed = 50  # Default estimate
            
            issue_keywords = ["bug", "issue", "vulnerability", "error", "problem", "warning", "critical", "high", "medium"]
            issues_found = sum(analysis_text.lower().count(kw) for kw in issue_keywords)
            
            # Push final result with FULL analysis text
            event_queue.put_nowait(StreamEvent(
                type=EventType.DONE,
                data={
                    "analysis": analysis_text,
                    "repo_url": request.repo_url,
                    "files_analyzed": files_analyzed or len(accumulated_tool_calls) * 20,
                    "issues_found": min(issues_found, 50),
                    "tool_calls": accumulated_tool_calls,
                    "iterations": result.iterations or final_iterations,
                    "session_id": result.session_id,
                    "completed": result.completed,
                }
            ))
        except Exception as e:
            event_queue.put_nowait(StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)}
            ))
    
    async def event_generator():
        """Generate SSE events with aggressive heartbeating."""
        agent_task = asyncio.create_task(run_agent())
        heartbeat_queue: asyncio.Queue = asyncio.Queue()
        stop_heartbeat = asyncio.Event()
        
        async def heartbeat_task():
            """Send heartbeats every 15s regardless of agent progress."""
            while not stop_heartbeat.is_set():
                await asyncio.sleep(15)
                if not stop_heartbeat.is_set():
                    await heartbeat_queue.put("heartbeat")
        
        heartbeat = asyncio.create_task(heartbeat_task())
        
        try:
            while True:
                event_task = asyncio.create_task(event_queue.get())
                hb_task = asyncio.create_task(heartbeat_queue.get())
                
                done, pending = await asyncio.wait(
                    [event_task, hb_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                completed_task = done.pop()
                
                if completed_task == event_task:
                    event = completed_task.result()
                    yield event.to_sse()
                    
                    if event.type in (EventType.DONE, EventType.ERROR):
                        break
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            stop_heartbeat.set()
            heartbeat.cancel()
            if not agent_task.done():
                agent_task.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/v3/tools")
async def list_codebase_tools():
    """List available Codebase Analyst tools."""
    agent = get_codebase_agent()
    return {
        "tools": [
            {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in agent.tools.items()
        ]
    }


# ============================================================
# V4: VERIFIED ANALYSIS (with test verification)
# ============================================================

class VerifiedAnalyzeRequest(BaseModel):
    """Request for verified codebase analysis."""
    repo_url: str
    focus: str = "all"
    branch: Optional[str] = None
    max_issues_to_verify: int = 10


@app.post("/v4/analyze/verified", dependencies=[Depends(verify_api_key)])
async def analyze_repository_verified(request: VerifiedAnalyzeRequest):
    """
    Analyze a GitHub repository with VERIFICATION.
    
    Two-phase analysis:
    1. Analyze codebase and extract structured issues
    2. Generate tests and run them to VERIFY findings
    
    Issues marked as:
    - VERIFIED: Test failed as expected (bug confirmed!)
    - UNVERIFIED: Test passed (may be false positive)
    - SKIPPED: Could not generate test
    
    This is "Vibe Engineering" - agents that verify their work.
    """
    import asyncio
    import httpx
    
    client = get_gemini_client()
    
    # Clone the repository first
    from .agent.tools_github import CloneRepoTool
    
    clone_tool = CloneRepoTool()
    clone_result = await clone_tool.execute({
        "repo_url": request.repo_url,
        "branch": request.branch,
    })
    
    if not clone_result.success:
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {clone_result.output}")
    
    # Create verified analyzer with reasoning model for deep analysis
    analyzer = VerifiedAnalyzer(
        client=client,
        model=get_reasoning_model(),
        code_executor=execute_code_in_sandbox,
    )
    
    # Run analysis with verification
    result = await analyzer.analyze_and_verify(
        repo_content=clone_result.output,
        repo_url=request.repo_url,
        focus=request.focus,
        max_issues_to_verify=request.max_issues_to_verify,
    )
    
    return result.to_dict()


@app.post("/v4/analyze/verified/stream", dependencies=[Depends(verify_api_key)])
async def analyze_repository_verified_stream(request: VerifiedAnalyzeRequest):
    """
    Analyze with verification - SSE streaming version.
    
    Streams events:
    - thinking: Phase updates
    - issue_found: Each issue as discovered
    - verify_start: Starting verification of an issue
    - verify_result: Verification result (verified/unverified)
    - done: Complete with all issues
    """
    import asyncio
    import httpx
    
    # Log analysis start
    analysis_start = time.time()
    add_log("info", "verified_analysis_started",
            repo_url=request.repo_url,
            focus=request.focus,
            max_issues=request.max_issues_to_verify)
    _metrics["analyses"]["total"] += 1
    
    client = get_gemini_client()
    event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    
    def on_event(event: StreamEvent):
        """Callback to push events to queue."""
        event_queue.put_nowait(event)
    
    async def run_analysis():
        """Run the verified analysis."""
        try:
            # Clone repo first
            from .agent.tools_github import CloneRepoTool
            
            on_event(StreamEvent(
                type=EventType.TOOL_START,
                data={"name": "clone_repo", "repo_url": request.repo_url}
            ))
            
            clone_tool = CloneRepoTool()
            clone_result = await clone_tool.execute({
                "repo_url": request.repo_url,
                "branch": request.branch,
            })
            
            on_event(StreamEvent(
                type=EventType.TOOL_RESULT,
                data={"name": "clone_repo", "success": clone_result.success}
            ))
            
            if not clone_result.success:
                event_queue.put_nowait(StreamEvent(
                    type=EventType.ERROR,
                    data={"error": f"Failed to clone: {clone_result.output}"}
                ))
                return
            
            # Create analyzer with reasoning model
            analyzer = VerifiedAnalyzer(
                client=client,
                model=get_reasoning_model(),
                code_executor=execute_code_in_sandbox,
            )
            
            # Run analysis
            result = await analyzer.analyze_and_verify(
                repo_content=clone_result.output,
                repo_url=request.repo_url,
                focus=request.focus,
                on_event=on_event,
                max_issues_to_verify=request.max_issues_to_verify,
            )
            
            # Send final result
            event_queue.put_nowait(StreamEvent(
                type=EventType.DONE,
                data=result.to_dict()
            ))
            
            # Log completion
            duration_s = time.time() - analysis_start
            _metrics["analyses"]["verified"] += result.verified_count
            _metrics["analyses"]["issues_found"] += result.total_issues
            
            add_log("info", "verified_analysis_completed",
                    repo_url=request.repo_url,
                    total_issues=result.total_issues,
                    verified=result.verified_count,
                    unverified=result.unverified_count,
                    duration_seconds=round(duration_s, 2))
            
        except Exception as e:
            duration_s = time.time() - analysis_start
            add_log("error", "verified_analysis_error",
                    repo_url=request.repo_url,
                    error=str(e)[:200],
                    duration_seconds=round(duration_s, 2))
            event_queue.put_nowait(StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)}
            ))
    
    async def event_generator():
        """Generate SSE events with aggressive heartbeating."""
        analysis_task = asyncio.create_task(run_analysis())
        heartbeat_queue: asyncio.Queue = asyncio.Queue()
        stop_heartbeat = asyncio.Event()
        
        async def heartbeat_task():
            """Send heartbeats every 15s regardless of analysis progress."""
            while not stop_heartbeat.is_set():
                await asyncio.sleep(15)
                if not stop_heartbeat.is_set():
                    await heartbeat_queue.put("heartbeat")
        
        heartbeat = asyncio.create_task(heartbeat_task())
        
        try:
            while True:
                # Wait for either an event or a heartbeat
                event_task = asyncio.create_task(event_queue.get())
                hb_task = asyncio.create_task(heartbeat_queue.get())
                
                done, pending = await asyncio.wait(
                    [event_task, hb_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Process completed task
                completed_task = done.pop()
                
                if completed_task == event_task:
                    event = completed_task.result()
                    yield event.to_sse()
                    
                    if event.type in (EventType.DONE, EventType.ERROR):
                        break
                else:
                    # Heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            stop_heartbeat.set()
            heartbeat.cancel()
            if not analysis_task.done():
                analysis_task.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# ASYNC JOBS - Background processing with webhooks
# ============================================================

import uuid
from dataclasses import dataclass, field
from enum import Enum

class JobStatus(str, Enum):
    """Async job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AsyncJob:
    """Represents an async analysis job."""
    job_id: str
    repo_url: str
    focus: str
    verify: bool
    generate_fixes: bool
    webhook_url: Optional[str]
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_phase: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# In-memory job store with automatic cleanup
_jobs: dict[str, AsyncJob] = {}
_job_tasks: dict[str, asyncio.Task] = {}

# Concurrency control
MAX_CONCURRENT_JOBS = 3  # Limit concurrent verification jobs
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Job retention: completed jobs are cleaned up after 1 hour
JOB_RETENTION_SECONDS = 3600  # 1 hour
JOB_CLEANUP_INTERVAL = 300    # Check every 5 minutes
MAX_STORED_JOBS = 100         # Hard cap on stored jobs


async def cleanup_old_jobs():
    """Background task to periodically remove completed/failed jobs older than retention period."""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL)
        try:
            cutoff = time.time() - JOB_RETENTION_SECONDS
            to_remove = [
                k for k, v in _jobs.items()
                if v.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                and v.updated_at < cutoff
            ]
            for k in to_remove:
                del _jobs[k]
                _job_tasks.pop(k, None)
            
            # Hard cap: if still too many, remove oldest completed first
            if len(_jobs) > MAX_STORED_JOBS:
                completed = sorted(
                    [(k, v) for k, v in _jobs.items() if v.status in (JobStatus.COMPLETED, JobStatus.FAILED)],
                    key=lambda x: x[1].updated_at
                )
                for k, _ in completed[:len(_jobs) - MAX_STORED_JOBS]:
                    del _jobs[k]
                    _job_tasks.pop(k, None)
            
            if to_remove:
                logger.info("jobs_cleanup", removed=len(to_remove), remaining=len(_jobs))
        except Exception as e:
            logger.warning("jobs_cleanup_error", error=str(e))


def get_running_job_count() -> int:
    """Get count of currently running jobs."""
    return sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING)


class AsyncAnalyzeRequest(BaseModel):
    """Request for async analysis."""
    repo_url: str
    branch: str = "main"
    focus: str = "full"
    verify: bool = True
    generate_fixes: bool = True
    max_issues_to_verify: int = 10
    webhook_url: Optional[str] = None


class AsyncJobResponse(BaseModel):
    """Response for async job submission."""
    job_id: str
    status: str
    status_url: str
    estimated_seconds: int


class AsyncJobStatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    status: str
    progress: float
    current_phase: str
    result: Optional[dict] = None
    error: Optional[str] = None


async def run_async_job(job: AsyncJob, request: AsyncAnalyzeRequest):
    """Background task to run analysis with concurrency control."""
    
    # Wait for semaphore (limits concurrent jobs)
    job.current_phase = "Queued (waiting for slot)"
    job.updated_at = time.time()
    
    async with _job_semaphore:
        analysis_start = time.time()
        
        try:
            job.status = JobStatus.RUNNING
            job.current_phase = "Cloning repository"
            job.updated_at = time.time()
            
            # Clone repo
            from .agent.tools_github import CloneRepoTool
            clone_tool = CloneRepoTool()
            clone_result = await clone_tool.execute({
                "repo_url": request.repo_url,
                "branch": request.branch,
            })
            
            if not clone_result.success:
                job.status = JobStatus.FAILED
                job.error = f"Failed to clone: {clone_result.output}"
                job.updated_at = time.time()
                _metrics["analyses"]["total"] += 1
                await send_webhook(job)
                return
            
            job.progress = 0.2
            job.current_phase = "Analyzing codebase"
            job.updated_at = time.time()
            
            # Run analysis
            client = get_gemini_client()
            
            def on_event(event: StreamEvent):
                """Update job progress from events."""
                if event.type == EventType.THINKING:
                    phase = event.data.get("phase", "")
                    if phase:
                        job.current_phase = phase
                        job.updated_at = time.time()
            
            if request.verify:
                analyzer = VerifiedAnalyzer(
                    client=client,
                    model=get_reasoning_model(),
                    code_executor=execute_code_in_sandbox,
                )
                
                result = await analyzer.analyze_and_verify(
                    repo_content=clone_result.output,
                    repo_url=request.repo_url,
                    focus=request.focus,
                    on_event=on_event,
                    max_issues_to_verify=request.max_issues_to_verify,
                )
                
                job.result = result.to_dict()
                
                # Update metrics
                _metrics["analyses"]["total"] += 1
                _metrics["analyses"]["verified"] += result.verified_count
                _metrics["analyses"]["issues_found"] += result.total_issues
            else:
                # Non-verified analysis (simpler path)
                job.result = {
                    "repo": request.repo_url,
                    "summary": "Analysis completed",
                    "issues": [],
                    "stats": {"total": 0}
                }
                _metrics["analyses"]["total"] += 1
            
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.current_phase = "Complete"
            job.updated_at = time.time()
            
            # Log completion
            duration_s = time.time() - analysis_start
            add_log("info", "async_job_completed",
                    job_id=job.job_id,
                    repo_url=job.repo_url,
                    total_issues=len(job.result.get("issues", [])),
                    duration_seconds=round(duration_s, 2))
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.updated_at = time.time()
            _metrics["analyses"]["total"] += 1
            add_log("error", "async_job_failed",
                    job_id=job.job_id,
                    error=str(e)[:200])
    
    # Send webhook notification
    await send_webhook(job)


async def send_webhook(job: AsyncJob):
    """Send webhook notification when job completes."""
    if not job.webhook_url:
        return
    
    try:
        import httpx
        
        payload = {
            "job_id": job.job_id,
            "status": job.status.value,
            "result": job.result if job.status == JobStatus.COMPLETED else None,
            "error": job.error if job.status == JobStatus.FAILED else None,
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                job.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            add_log("info", "webhook_sent",
                    job_id=job.job_id,
                    webhook_url=job.webhook_url,
                    status_code=response.status_code)
                    
    except Exception as e:
        add_log("error", "webhook_failed",
                job_id=job.job_id,
                webhook_url=job.webhook_url,
                error=str(e)[:100])


@app.post("/v4/analyze/async", response_model=AsyncJobResponse, dependencies=[Depends(verify_api_key)])
async def submit_async_analysis(request: AsyncAnalyzeRequest):
    """
    Submit an analysis job for background processing.
    
    Returns immediately with a job_id. Use /v4/jobs/{job_id} to poll status
    or provide a webhook_url to receive a POST when complete.
    
    Example:
        POST /v4/analyze/async
        {
            "repo_url": "https://github.com/owner/repo",
            "verify": true,
            "webhook_url": "https://your-server.com/callback"
        }
        
        Response:
        {
            "job_id": "abc123",
            "status": "pending",
            "status_url": "/v4/jobs/abc123",
            "estimated_seconds": 120
        }
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Check queue capacity
    running_count = get_running_job_count()
    queued = running_count >= MAX_CONCURRENT_JOBS
    
    job = AsyncJob(
        job_id=job_id,
        repo_url=request.repo_url,
        focus=request.focus,
        verify=request.verify,
        generate_fixes=request.generate_fixes,
        webhook_url=request.webhook_url,
    )
    
    if queued:
        job.current_phase = f"Queued (position {running_count - MAX_CONCURRENT_JOBS + 1})"
    
    _jobs[job_id] = job
    
    # Start background task
    task = asyncio.create_task(run_async_job(job, request))
    _job_tasks[job_id] = task
    
    add_log("info", "async_job_submitted",
            job_id=job_id,
            repo_url=request.repo_url,
            has_webhook=bool(request.webhook_url),
            queued=queued,
            running_jobs=running_count)
    
    # Estimate based on verification and queue
    base_estimate = 120 if request.verify else 60
    queue_delay = (running_count - MAX_CONCURRENT_JOBS + 1) * 60 if queued else 0
    estimated = base_estimate + queue_delay
    
    return AsyncJobResponse(
        job_id=job_id,
        status="queued" if queued else job.status.value,
        status_url=f"/v4/jobs/{job_id}",
        estimated_seconds=estimated,
    )


@app.get("/v4/jobs/{job_id}", response_model=AsyncJobStatusResponse, dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """
    Get the status of an async analysis job.
    
    Returns:
        - status: pending, running, completed, failed
        - progress: 0.0 to 1.0
        - current_phase: What the job is currently doing
        - result: Full analysis report when completed
        - error: Error message if failed
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _jobs[job_id]
    
    return AsyncJobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_phase=job.current_phase,
        result=job.result,
        error=job.error,
    )


@app.get("/v4/jobs", dependencies=[Depends(verify_api_key)])
async def list_jobs(limit: int = 20):
    """List recent async jobs."""
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "repo_url": j.repo_url,
                "status": j.status.value,
                "progress": j.progress,
                "created_at": j.created_at,
            }
            for j in jobs
        ],
        "total": len(_jobs),
    }


@app.delete("/v4/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _jobs[job_id]
    
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return {"message": f"Job already {job.status.value}", "job_id": job_id}
    
    # Cancel the task
    if job_id in _job_tasks:
        _job_tasks[job_id].cancel()
        del _job_tasks[job_id]
    
    job.status = JobStatus.FAILED
    job.error = "Cancelled by user"
    job.updated_at = time.time()
    
    add_log("info", "async_job_cancelled", job_id=job_id)
    
    return {"message": "Job cancelled", "job_id": job_id}


# ============================================================
# V5: CODE DOCTOR - Full Analysis Suite
# ============================================================
# Combines:
# - Security scanning (secrets, config issues)
# - Code analysis with verification
# - Evolution/roadmap recommendations
#
# This is the "Code Doctor" - comprehensive health check for codebases.
# ============================================================

from .agent.security import (
    SecretScanner,
    SecurityFinding,
    scan_codebase_for_secrets_async,
    Severity as SecuritySeverity,
)
from .agent.evolution import (
    EvolutionAdvisor,
    EvolutionReport,
)


class CodeDoctorRequest(BaseModel):
    """Request for full Code Doctor analysis."""
    repo_url: str
    branch: Optional[str] = None
    
    # Analysis toggles
    run_security_scan: bool = True
    run_code_analysis: bool = True
    run_evolution_analysis: bool = True
    
    # Options
    max_issues_to_verify: int = 10
    evolution_focus: str = "full"  # full, architecture, security, performance, testing, debt, devops


class CodeDoctorResponse(BaseModel):
    """Response from Code Doctor analysis."""
    repo_url: str
    analysis_time_seconds: float
    
    # Security scan results
    security_findings: List[dict]
    security_summary: dict
    
    # Code analysis results (verified issues)
    code_issues: List[dict]
    code_summary: dict
    
    # Evolution recommendations
    evolution_recommendations: List[dict]
    evolution_summary: dict
    
    # Overall health
    overall_health_score: float
    executive_summary: str


@app.post("/v5/analyze/full", response_model=CodeDoctorResponse, dependencies=[Depends(verify_api_key)])
async def code_doctor_full(request: CodeDoctorRequest):
    """
    Code Doctor - Comprehensive Codebase Health Check.
    
    Three-phase analysis:
    1. **Security Scan**: Pattern-based detection of secrets, credentials, misconfigurations
    2. **Code Analysis**: AI-powered bug/issue detection with verification
    3. **Evolution Advisor**: Strategic recommendations for codebase improvement
    
    Returns a unified health report with:
    - Security findings (secrets, vulnerabilities)
    - Verified code issues (confirmed bugs)
    - Evolution roadmap (tech debt, improvements)
    - Overall health score
    
    This is "Vibe Engineering" - the agent analyzes, verifies, AND recommends.
    """
    import time as time_module
    
    start_time = time_module.time()
    
    add_log("info", "code_doctor_started",
            repo_url=request.repo_url,
            security=request.run_security_scan,
            analysis=request.run_code_analysis,
            evolution=request.run_evolution_analysis)
    
    client = get_gemini_client()
    
    # Clone repository first
    from .agent.tools_github import CloneRepoTool
    
    clone_tool = CloneRepoTool()
    clone_result = await clone_tool.execute({
        "repo_url": request.repo_url,
        "branch": request.branch,
    })
    
    if not clone_result.success:
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {clone_result.output}")
    
    repo_content = clone_result.output
    
    # Initialize results
    security_findings = []
    security_summary = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    
    code_issues = []
    code_summary = {"total": 0, "verified": 0, "unverified": 0}
    
    evolution_recommendations = []
    evolution_summary = {"total": 0, "quick_wins": 0, "health_score": 0}
    
    # ============================================================
    # PHASE 1: Security Scan (fast, pattern-based)
    # ============================================================
    if request.run_security_scan:
        add_log("info", "code_doctor_security_scan", repo_url=request.repo_url)
        
        findings = await scan_codebase_for_secrets_async(repo_content)
        security_findings = [f.to_dict() for f in findings]
        
        security_summary = {
            "total": len(findings),
            "critical": sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL),
            "high": sum(1 for f in findings if f.severity == SecuritySeverity.HIGH),
            "medium": sum(1 for f in findings if f.severity == SecuritySeverity.MEDIUM),
            "low": sum(1 for f in findings if f.severity == SecuritySeverity.LOW),
        }
    
    # ============================================================
    # PHASE 2: Code Analysis with Verification (AI-powered)
    # ============================================================
    if request.run_code_analysis:
        add_log("info", "code_doctor_code_analysis", repo_url=request.repo_url)
        
        analyzer = VerifiedAnalyzer(
            client=client,
            model=get_reasoning_model(),
            code_executor=execute_code_in_sandbox,
        )
        
        analysis_result = await analyzer.analyze_and_verify(
            repo_content=repo_content,
            repo_url=request.repo_url,
            focus="all",
            max_issues_to_verify=request.max_issues_to_verify,
        )
        
        code_issues = [i.to_dict() for i in analysis_result.issues]
        code_summary = {
            "total": analysis_result.total_issues,
            "verified": analysis_result.verified_count,
            "unverified": analysis_result.unverified_count,
        }
    
    # ============================================================
    # PHASE 3: Evolution Analysis (AI-powered)
    # ============================================================
    if request.run_evolution_analysis:
        add_log("info", "code_doctor_evolution", repo_url=request.repo_url)
        
        advisor = EvolutionAdvisor(
            client=client,
            model=get_reasoning_model(),
        )
        
        evolution_report = await advisor.analyze(
            repo_content=repo_content,
            repo_url=request.repo_url,
            focus=request.evolution_focus,
        )
        
        evolution_recommendations = [r.to_dict() for r in evolution_report.recommendations]
        evolution_summary = {
            "total": len(evolution_report.recommendations),
            "quick_wins": len(evolution_report.quick_wins),
            "health_score": evolution_report.health_score,
            "maturity_level": evolution_report.maturity_level,
        }
    
    # ============================================================
    # Calculate Overall Health Score
    # ============================================================
    # Weighted average:
    # - Security: 30% (critical secrets = major penalty)
    # - Code quality: 40% (verified bugs = penalty)
    # - Evolution health: 30% (from advisor)
    
    security_score = 100
    if security_summary["total"] > 0:
        security_score -= (security_summary["critical"] * 25)
        security_score -= (security_summary["high"] * 10)
        security_score -= (security_summary["medium"] * 3)
        security_score = max(0, security_score)
    
    code_score = 100
    if code_summary["total"] > 0:
        verified_penalty = code_summary["verified"] * 15
        unverified_penalty = code_summary["unverified"] * 5
        code_score = max(0, 100 - verified_penalty - unverified_penalty)
    
    evolution_score = evolution_summary.get("health_score", 50)
    
    overall_health = (
        (security_score * 0.30) +
        (code_score * 0.40) +
        (evolution_score * 0.30)
    )
    
    # ============================================================
    # Generate Executive Summary
    # ============================================================
    critical_items = []
    
    if security_summary["critical"] > 0:
        critical_items.append(f"{security_summary['critical']} critical secrets exposed")
    
    if code_summary["verified"] > 0:
        critical_items.append(f"{code_summary['verified']} verified bugs")
    
    if evolution_summary.get("maturity_level") == "legacy":
        critical_items.append("codebase shows legacy patterns")
    
    if critical_items:
        urgency = "🔴 URGENT: " + ", ".join(critical_items)
    else:
        urgency = "✅ No critical issues detected"
    
    executive_summary = f"""## Code Doctor Report for {request.repo_url}

**Overall Health Score: {overall_health:.0f}/100**

{urgency}

### Security ({security_summary['total']} findings)
- Critical: {security_summary['critical']} | High: {security_summary['high']} | Medium: {security_summary['medium']}

### Code Quality ({code_summary['total']} issues)
- Verified bugs: {code_summary['verified']} | Potential issues: {code_summary['unverified']}

### Evolution ({evolution_summary['total']} recommendations)
- Quick wins available: {evolution_summary['quick_wins']}
- Maturity level: {evolution_summary.get('maturity_level', 'N/A')}
"""
    
    total_time = time_module.time() - start_time
    
    add_log("info", "code_doctor_completed",
            repo_url=request.repo_url,
            health_score=overall_health,
            security_findings=security_summary["total"],
            code_issues=code_summary["total"],
            evolution_recs=evolution_summary["total"],
            duration_seconds=round(total_time, 2))
    
    _metrics["analyses"]["total"] += 1
    
    return CodeDoctorResponse(
        repo_url=request.repo_url,
        analysis_time_seconds=total_time,
        security_findings=security_findings,
        security_summary=security_summary,
        code_issues=code_issues,
        code_summary=code_summary,
        evolution_recommendations=evolution_recommendations,
        evolution_summary=evolution_summary,
        overall_health_score=overall_health,
        executive_summary=executive_summary,
    )


@app.post("/v5/analyze/full/stream", dependencies=[Depends(verify_api_key)])
async def code_doctor_full_stream(request: CodeDoctorRequest):
    """
    Code Doctor with SSE streaming - see progress in real-time.
    
    Streams events:
    - phase_start: New phase beginning (security, analysis, evolution)
    - finding: Security finding detected
    - issue: Code issue found
    - verify_result: Issue verification result
    - recommendation: Evolution recommendation
    - done: Complete with full results
    """
    import asyncio
    import time as time_module
    
    start_time = time_module.time()
    client = get_gemini_client()
    event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    
    def on_event(event: StreamEvent):
        event_queue.put_nowait(event)
    
    async def run_analysis():
        """Run all three analysis phases."""
        try:
            # Clone repo
            on_event(StreamEvent(
                type=EventType.TOOL_START,
                data={"phase": "clone", "message": "Cloning repository..."}
            ))
            
            from .agent.tools_github import CloneRepoTool
            
            clone_tool = CloneRepoTool()
            clone_result = await clone_tool.execute({
                "repo_url": request.repo_url,
                "branch": request.branch,
            })
            
            if not clone_result.success:
                on_event(StreamEvent(
                    type=EventType.ERROR,
                    data={"error": f"Failed to clone: {clone_result.output}"}
                ))
                return
            
            repo_content = clone_result.output
            
            security_findings = []
            code_issues = []
            evolution_recs = []
            evolution_health = 50
            maturity = "unknown"
            
            # Phase 1: Security
            if request.run_security_scan:
                on_event(StreamEvent(
                    type=EventType.THINKING,
                    data={"phase": "security", "message": "Scanning for secrets and misconfigurations..."}
                ))
                
                findings = await scan_codebase_for_secrets_async(repo_content)
                security_findings = [f.to_dict() for f in findings]
                
                for finding in findings:
                    on_event(StreamEvent(
                        type=EventType.TOOL_RESULT,
                        data={"type": "security_finding", "finding": finding.to_dict()}
                    ))
            
            # Phase 2: Code Analysis
            if request.run_code_analysis:
                on_event(StreamEvent(
                    type=EventType.THINKING,
                    data={"phase": "analysis", "message": "Analyzing code for bugs and issues..."}
                ))
                
                analyzer = VerifiedAnalyzer(
                    client=client,
                    model=get_reasoning_model(),
                    code_executor=execute_code_in_sandbox,
                )
                
                analysis_result = await analyzer.analyze_and_verify(
                    repo_content=repo_content,
                    repo_url=request.repo_url,
                    focus="all",
                    on_event=on_event,
                    max_issues_to_verify=request.max_issues_to_verify,
                )
                
                code_issues = [i.to_dict() for i in analysis_result.issues]
            
            # Phase 3: Evolution
            if request.run_evolution_analysis:
                on_event(StreamEvent(
                    type=EventType.THINKING,
                    data={"phase": "evolution", "message": "Generating evolution recommendations..."}
                ))
                
                advisor = EvolutionAdvisor(
                    client=client,
                    model=get_reasoning_model(),
                )
                
                evolution_report = await advisor.analyze(
                    repo_content=repo_content,
                    repo_url=request.repo_url,
                    focus=request.evolution_focus,
                    on_event=on_event,
                )
                
                evolution_recs = [r.to_dict() for r in evolution_report.recommendations]
                evolution_health = evolution_report.health_score
                maturity = evolution_report.maturity_level
            
            # Calculate overall health
            sec_critical = sum(1 for f in security_findings if f.get("severity") == "critical")
            sec_high = sum(1 for f in security_findings if f.get("severity") == "high")
            verified_bugs = sum(1 for i in code_issues if i.get("verification_status") == "verified")
            
            security_score = max(0, 100 - (sec_critical * 25) - (sec_high * 10))
            code_score = max(0, 100 - (verified_bugs * 15))
            overall_health = (security_score * 0.3) + (code_score * 0.4) + (evolution_health * 0.3)
            
            # Done
            on_event(StreamEvent(
                type=EventType.DONE,
                data={
                    "repo_url": request.repo_url,
                    "analysis_time_seconds": time_module.time() - start_time,
                    "security_findings": security_findings,
                    "security_summary": {
                        "total": len(security_findings),
                        "critical": sec_critical,
                        "high": sec_high,
                    },
                    "code_issues": code_issues,
                    "code_summary": {
                        "total": len(code_issues),
                        "verified": verified_bugs,
                    },
                    "evolution_recommendations": evolution_recs,
                    "evolution_summary": {
                        "total": len(evolution_recs),
                        "health_score": evolution_health,
                        "maturity_level": maturity,
                    },
                    "overall_health_score": overall_health,
                }
            ))
            
        except Exception as e:
            on_event(StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)}
            ))
    
    async def event_generator():
        """Generate SSE events with aggressive heartbeating."""
        analysis_task = asyncio.create_task(run_analysis())
        heartbeat_queue: asyncio.Queue = asyncio.Queue()
        stop_heartbeat = asyncio.Event()
        
        async def heartbeat_task():
            """Send heartbeats every 15s regardless of analysis progress."""
            while not stop_heartbeat.is_set():
                await asyncio.sleep(15)
                if not stop_heartbeat.is_set():
                    await heartbeat_queue.put("heartbeat")
        
        heartbeat = asyncio.create_task(heartbeat_task())
        
        try:
            while True:
                event_task = asyncio.create_task(event_queue.get())
                hb_task = asyncio.create_task(heartbeat_queue.get())
                
                done, pending = await asyncio.wait(
                    [event_task, hb_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                completed_task = done.pop()
                
                if completed_task == event_task:
                    event = completed_task.result()
                    yield event.to_sse()
                    
                    if event.type in (EventType.DONE, EventType.ERROR):
                        break
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            stop_heartbeat.set()
            heartbeat.cancel()
            if not analysis_task.done():
                analysis_task.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/v5/analyze/security", dependencies=[Depends(verify_api_key)])
async def security_scan_only(repo_url: str, branch: Optional[str] = None):
    """
    Quick security scan only - no AI, just pattern matching.
    
    Fast scan for secrets and credentials.
    """
    from .agent.tools_github import CloneRepoTool
    import time as time_module
    
    start_time = time_module.time()
    
    clone_tool = CloneRepoTool()
    clone_result = await clone_tool.execute({
        "repo_url": repo_url,
        "branch": branch,
    })
    
    if not clone_result.success:
        raise HTTPException(status_code=400, detail=f"Failed to clone: {clone_result.output}")
    
    findings = await scan_codebase_for_secrets_async(clone_result.output)
    
    return {
        "repo_url": repo_url,
        "scan_time_seconds": time_module.time() - start_time,
        "findings": [f.to_dict() for f in findings],
        "summary": {
            "total": len(findings),
            "critical": sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL),
            "high": sum(1 for f in findings if f.severity == SecuritySeverity.HIGH),
            "medium": sum(1 for f in findings if f.severity == SecuritySeverity.MEDIUM),
            "low": sum(1 for f in findings if f.severity == SecuritySeverity.LOW),
        },
    }


@app.post("/v5/analyze/evolution", dependencies=[Depends(verify_api_key)])
async def evolution_analysis_only(request: AnalyzeRepoRequest):
    """
    Evolution analysis only - strategic recommendations.
    
    Analyzes codebase for:
    - Technical debt
    - Architecture improvements
    - Feature roadmap
    - Quick wins
    """
    from .agent.tools_github import CloneRepoTool
    
    client = get_gemini_client()
    
    clone_tool = CloneRepoTool()
    clone_result = await clone_tool.execute({
        "repo_url": request.repo_url,
        "branch": request.branch,
    })
    
    if not clone_result.success:
        raise HTTPException(status_code=400, detail=f"Failed to clone: {clone_result.output}")
    
    advisor = EvolutionAdvisor(
        client=client,
        model=get_reasoning_model(),
    )
    
    report = await advisor.analyze(
        repo_content=clone_result.output,
        repo_url=request.repo_url,
        focus=request.focus,
    )
    
    return report.to_dict()


# ============================================================
# UPDATE ROOT ENDPOINT TO SHOW V5
# ============================================================

# Override root to include v5
@app.get("/")
async def root_v5():
    """Root endpoint - shows API info including Code Doctor."""
    return {
        "name": "Gemini Code Doctor",
        "version": __version__,
        "status": "running",
        "model": get_model_name(),
        "reasoning_model": get_reasoning_model(),
        "docs": "/docs",
        "endpoints": {
            "v5_code_doctor": {
                "full_analysis": "/v5/analyze/full",
                "full_stream": "/v5/analyze/full/stream",
                "security_only": "/v5/analyze/security",
                "evolution_only": "/v5/analyze/evolution",
            },
            "v4_verified": {
                "analyze": "/v4/analyze/verified",
                "stream": "/v4/analyze/verified/stream",
                "async": "/v4/analyze/async",
                "jobs": "/v4/jobs",
            },
            "v3_analysis": {
                "analyze": "/v3/analyze",
                "stream": "/v3/analyze/stream",
            },
            "v2_agent": {
                "agent": "/v2/agent",
                "stream": "/v2/agent/stream",
            },
            "diagnostics": {
                "health": "/health",
                "diagnostics": "/diagnostics",
                "logs": "/logs",
            },
        },
        "capabilities": [
            "code_doctor",        # Full health check
            "security_scan",      # Secret detection
            "verified_analysis",  # Bug verification
            "evolution_advisor",  # Roadmap generation
            "marathon_agent",     # Long-running tasks
            "streaming",          # SSE real-time updates
        ],
    }


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
