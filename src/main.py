"""
Gemini Agent - Hackathon Project

A Gemini-powered agent with tool calling capabilities.

Run locally:
    uvicorn src.main:app --reload

Deploy to Railway:
    git push (auto-deploys if connected)
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

# Load environment variables
load_dotenv()

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
    return os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


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
        "name": "Gemini Agent",
        "version": "0.1.0",
        "status": "running",
        "model": get_model_name(),
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint for Railway/deployment platforms."""
    return {
        "status": "healthy",
        "model": get_model_name(),
    }


@app.post("/generate", response_model=GenerateResponse)
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


@app.post("/chat", response_model=GenerateResponse)
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


@app.post("/agent", response_model=AgentResponse)
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
    """List available tools."""
    return {"tools": TOOLS}


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
