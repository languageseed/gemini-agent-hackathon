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
"""

import os
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
    StreamEvent, EventType, EventCollector
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
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY environment variable not set"
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
        "version": "0.2.0",
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
        "version": "0.2.0",
        "capabilities": [
            "marathon_agent",
            "tool_calling",
            "code_execution",
            "session_persistence",
            "streaming",
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
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
