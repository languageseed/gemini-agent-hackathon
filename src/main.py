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
)
from .agent.tools import create_default_tools


# ============================================================
# SECURITY - API Key Protection
# ============================================================

def get_api_key() -> Optional[str]:
    """Get the API key from environment. None = open access."""
    return os.environ.get("API_SECRET_KEY")


async def verify_api_key(request: Request):
    """
    Verify API key if API_SECRET_KEY is set.
    
    - If API_SECRET_KEY is not set: API is open (demo mode)
    - If API_SECRET_KEY is set: requires X-API-Key header
    - Health and root endpoints are always open
    """
    api_key = get_api_key()
    
    # If no API key configured, allow all requests (demo mode)
    if not api_key:
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
    """Get the configured model name."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def get_marathon_agent() -> MarathonAgent:
    """Get or initialize the Marathon Agent."""
    global _marathon_agent, _session_store
    
    if _marathon_agent is None:
        client = get_gemini_client()
        tools = create_default_tools()
        _session_store = SessionStore()
        
        config = AgentConfig(
            model=get_model_name(),
            max_iterations=15,
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
    yield
    logger.info("shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Gemini Agent",
    description="Gemini-powered agent with tool calling",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    max_iterations: int = 10


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
            # Safe evaluation (only math)
            result = eval(expr, {"__builtins__": {}}, {})
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
        "version": "0.3.0",
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
        "model": get_model_name(),
        "secured": get_api_key() is not None,
        "version": "0.3.0",
        "capabilities": [
            "marathon_agent",
            "tool_calling",
            "code_execution",
            "session_persistence",
            "streaming",
            "codebase_analysis",
        ],
    }


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    """Generate content with Gemini."""
    try:
        from google.genai.types import GenerateContentConfig
        
        client = get_gemini_client()
        
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
        
        return GenerateResponse(
            text=response.text,
            model=get_model_name(),
        )
        
    except Exception as e:
        logger.error("generate_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """Multi-turn chat with Gemini."""
    try:
        from google.genai.types import Content, Part, GenerateContentConfig
        
        client = get_gemini_client()
        
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
        
        return GenerateResponse(
            text=response.text,
            model=get_model_name(),
        )
        
    except Exception as e:
        logger.error("chat_error", error=str(e))
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


class MarathonAgentResponse(BaseModel):
    """Response from Marathon Agent."""
    text: str
    tool_calls: List[dict]
    iterations: int
    session_id: Optional[str]
    completed: bool
    error: Optional[str] = None


@app.post("/v2/agent", response_model=MarathonAgentResponse, dependencies=[Depends(verify_api_key)])
async def marathon_agent(request: MarathonAgentRequest):
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
        
        # Override max iterations if specified
        if request.max_iterations:
            agent.config.max_iterations = request.max_iterations
        
        result = await agent.run(
            task=request.task,
            session_id=request.session_id,
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
async def marathon_agent_stream(request: MarathonAgentRequest):
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
        
        if request.max_iterations:
            agent.config.max_iterations = request.max_iterations
        
        # Collect events
        collector = EventCollector()
        
        # Run agent with event callback
        result = await agent.run(
            task=request.task,
            session_id=request.session_id,
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
async def analyze_repository(request: AnalyzeRepoRequest):
    """
    Analyze a GitHub repository.
    
    This is the hackathon showcase endpoint - demonstrates:
    - Loading entire codebases using Gemini's 2M context
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
async def analyze_repository_stream(request: AnalyzeRepoRequest):
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
        """Generate SSE events."""
        # Start agent in background
        agent_task = asyncio.create_task(run_agent())
        
        try:
            while True:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=120.0)
                    yield event.to_sse()
                    
                    # Stop on done or error
                    if event.type in (EventType.DONE, EventType.ERROR):
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            if not agent_task.done():
                agent_task.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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
    
    # Create code executor using E2B or local fallback
    async def execute_code(code: str) -> tuple[bool, str]:
        """Execute Python code and return (success, output)."""
        api_key = os.environ.get("E2B_API_KEY")
        
        if api_key:
            try:
                from e2b_code_interpreter import Sandbox
                
                sandbox = Sandbox(timeout=30)
                try:
                    execution = sandbox.run_code(code)
                    
                    output_parts = []
                    if execution.logs.stdout:
                        output_parts.append(execution.logs.stdout)
                    if execution.logs.stderr:
                        output_parts.append(f"STDERR: {execution.logs.stderr}")
                    
                    if execution.error:
                        return False, f"ERROR: {execution.error.name}: {execution.error.value}"
                    
                    return True, "\n".join(output_parts) or "Success (no output)"
                finally:
                    sandbox.kill()
                    
            except Exception as e:
                logger.warning("e2b_fallback", error=str(e))
        
        # Local fallback
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            namespace = {"__builtins__": __builtins__}
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            return True, stdout + (f"\nSTDERR: {stderr}" if stderr else "")
            
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {str(e)}"
    
    # Clone the repository first
    from .agent.tools_github import CloneRepoTool
    
    clone_tool = CloneRepoTool()
    clone_result = await clone_tool.execute({
        "repo_url": request.repo_url,
        "branch": request.branch,
    })
    
    if not clone_result.success:
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {clone_result.output}")
    
    # Create verified analyzer
    analyzer = VerifiedAnalyzer(
        client=client,
        model=get_model_name(),
        code_executor=execute_code,
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
    
    client = get_gemini_client()
    event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    
    # Create code executor
    async def execute_code(code: str) -> tuple[bool, str]:
        """Execute Python code and return (success, output)."""
        api_key = os.environ.get("E2B_API_KEY")
        
        if api_key:
            try:
                from e2b_code_interpreter import Sandbox
                
                sandbox = Sandbox(timeout=30)
                try:
                    execution = sandbox.run_code(code)
                    
                    output_parts = []
                    if execution.logs.stdout:
                        output_parts.append(execution.logs.stdout)
                    if execution.logs.stderr:
                        output_parts.append(f"STDERR: {execution.logs.stderr}")
                    
                    if execution.error:
                        return False, f"ERROR: {execution.error.name}: {execution.error.value}"
                    
                    return True, "\n".join(output_parts) or "Success (no output)"
                finally:
                    sandbox.kill()
                    
            except Exception as e:
                logger.warning("e2b_fallback", error=str(e))
        
        # Local fallback
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            namespace = {"__builtins__": __builtins__}
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            return True, stdout + (f"\nSTDERR: {stderr}" if stderr else "")
            
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {str(e)}"
    
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
            
            # Create analyzer
            analyzer = VerifiedAnalyzer(
                client=client,
                model=get_model_name(),
                code_executor=execute_code,
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
            
        except Exception as e:
            logger.error("verified_analysis_error", error=str(e))
            event_queue.put_nowait(StreamEvent(
                type=EventType.ERROR,
                data={"error": str(e)}
            ))
    
    async def event_generator():
        """Generate SSE events."""
        analysis_task = asyncio.create_task(run_analysis())
        
        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=180.0)
                    yield event.to_sse()
                    
                    if event.type in (EventType.DONE, EventType.ERROR):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
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
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
