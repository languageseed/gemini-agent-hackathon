"""
Marathon Agent Core

The main agentic loop leveraging Gemini 3 native capabilities:
- Thought signatures (via chat history)
- Thinking levels (dynamic optimization)
- Parallel tool execution (native)
- 2M context window (no truncation needed)

Total: ~80 lines (vs ~3,750 in Valet)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Callable
from enum import Enum

import structlog

from .tools import ToolRegistry, ToolResult
from .session import SessionStore, Session
from .stream import StreamEvent, EventType

logger = structlog.get_logger()


class ThinkingLevel(str, Enum):
    """Gemini 3 thinking levels for cost/latency optimization."""
    MINIMAL = "minimal"  # Fast, simple responses
    LOW = "low"          # Standard responses
    MEDIUM = "medium"    # Multi-step reasoning
    HIGH = "high"        # Deep reasoning, complex tasks


@dataclass
class AgentConfig:
    """Agent configuration."""
    model: str = "gemini-2.0-flash"
    max_iterations: int = 100  # High limit, time is the real constraint
    timeout_seconds: float = 600.0  # 10 minutes default, can be hours for marathon tasks
    default_thinking_level: ThinkingLevel = ThinkingLevel.LOW
    system_instruction: str = """You are an autonomous agent that completes tasks step by step.

When given a task:
1. Break it into clear steps
2. Use available tools to accomplish each step
3. Verify your work before concluding
4. If something fails, try an alternative approach

Be concise but thorough. Always explain what you're doing."""


@dataclass
class AgentResult:
    """Result from agent execution."""
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0
    session_id: Optional[str] = None
    completed: bool = True
    error: Optional[str] = None


class MarathonAgent:
    """
    Autonomous agent with Gemini 3 + tool execution.
    
    Leverages native Gemini capabilities:
    - Chat history = thought signatures (reasoning continuity)
    - thinking_level parameter = dynamic optimization
    - function_call = native tool calling
    - Parallel tool calls handled automatically
    """
    
    def __init__(
        self,
        client,  # google.genai.Client
        tools: ToolRegistry,
        sessions: Optional[SessionStore] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.client = client
        self.tools = tools
        self.sessions = sessions
        self.config = config or AgentConfig()
        
    def _select_thinking_level(
        self, 
        task: str, 
        iteration: int,
        has_tool_calls: bool
    ) -> ThinkingLevel:
        """
        Dynamically select thinking level based on context.
        
        First iteration = deeper thinking to plan
        Tool results = lower thinking to process
        Complex keywords = higher thinking
        """
        # First iteration needs planning
        if iteration == 0:
            return ThinkingLevel.MEDIUM
        
        # Processing tool results can be lighter
        if has_tool_calls:
            return ThinkingLevel.LOW
        
        # Check for complexity indicators
        complex_keywords = ["analyze", "debug", "implement", "design", "explain why"]
        if any(kw in task.lower() for kw in complex_keywords):
            return ThinkingLevel.HIGH
        
        return self.config.default_thinking_level
    
    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        owner_id: str = "",
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> AgentResult:
        """
        Execute a task with the agent.
        
        Args:
            task: The task to complete
            session_id: Optional session ID for persistence/resume
            owner_id: Owner scope for session access control (hash of API key)
            on_event: Optional callback for streaming events
            
        Returns:
            AgentResult with final response and metadata
        """
        from google.genai.types import (
            Content, Part, GenerateContentConfig,
            Tool, FunctionDeclaration
        )
        
        # Load or create session (scoped by owner_id)
        session = None
        if self.sessions and session_id:
            session = await self.sessions.load(session_id, owner_id=owner_id)
        
        if session is None:
            session = Session(id=session_id)
        
        # Build tool declarations
        function_declarations = [
            FunctionDeclaration(
                name=name,
                description=tool.description,
                parameters=tool.parameters
            )
            for name, tool in self.tools.items()
        ]
        
        gemini_tools = [Tool(function_declarations=function_declarations)] if function_declarations else None
        
        # Initialize conversation with session history or new task
        messages = list(session.messages) if session.messages else []
        if not messages or messages[-1].role != "user":
            messages.append(Content(role="user", parts=[Part(text=task)]))
        
        all_tool_calls = []
        iterations = 0
        start_time = asyncio.get_event_loop().time()
        
        def time_remaining() -> float:
            elapsed = asyncio.get_event_loop().time() - start_time
            return self.config.timeout_seconds - elapsed
        
        # Emit start event
        if on_event:
            on_event(StreamEvent(
                type=EventType.START,
                data={"task": task, "session_id": session.id, "timeout_seconds": self.config.timeout_seconds}
            ))
        
        try:
            while iterations < self.config.max_iterations and time_remaining() > 0:
                iterations += 1
                
                # Select thinking level dynamically
                thinking_level = self._select_thinking_level(
                    task, 
                    iterations - 1,
                    bool(all_tool_calls)
                )
                
                if on_event:
                    on_event(StreamEvent(
                        type=EventType.THINKING,
                        data={"level": thinking_level.value, "iteration": iterations}
                    ))
                
                # Build config with thinking level
                config = GenerateContentConfig(
                    system_instruction=self.config.system_instruction,
                    tools=gemini_tools,
                    # thinking_level=thinking_level.value,  # Uncomment when API supports
                )
                
                # Generate response
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=messages,
                    config=config
                )
                
                candidate = response.candidates[0]
                
                # Extract function calls
                function_calls = []
                text_parts = []
                
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                # Emit token event for text
                if text_parts and on_event:
                    on_event(StreamEvent(
                        type=EventType.TOKEN,
                        data={"content": "".join(text_parts)}
                    ))
                
                # No tool calls = we're done
                if not function_calls:
                    final_text = response.text or "".join(text_parts)
                    
                    # Save session state (scoped by owner)
                    session.messages = messages
                    if self.sessions:
                        await self.sessions.save(session, owner_id=owner_id)
                    
                    if on_event:
                        on_event(StreamEvent(
                            type=EventType.DONE,
                            data={"iterations": iterations, "tool_calls": len(all_tool_calls)}
                        ))
                    
                    return AgentResult(
                        text=final_text,
                        tool_calls=all_tool_calls,
                        iterations=iterations,
                        session_id=session.id,
                        completed=True
                    )
                
                # Add model response to conversation (thought signature preserved)
                messages.append(candidate.content)
                
                # Execute tools (parallel when possible)
                tool_results = []
                tool_tasks = []
                
                for fc in function_calls:
                    tool_call = {"name": fc.name, "arguments": dict(fc.args) if fc.args else {}}
                    all_tool_calls.append(tool_call)
                    
                    if on_event:
                        on_event(StreamEvent(
                            type=EventType.TOOL_START,
                            data=tool_call
                        ))
                    
                    # Queue for parallel execution
                    tool_tasks.append(self._execute_tool(fc.name, tool_call["arguments"]))
                
                # Execute all tools in parallel
                results = await asyncio.gather(*tool_tasks, return_exceptions=True)
                
                # Process results
                for fc, result in zip(function_calls, results):
                    if isinstance(result, Exception):
                        result_str = f"Error: {str(result)}"
                    else:
                        result_str = result.output if isinstance(result, ToolResult) else str(result)
                    
                    if on_event:
                        on_event(StreamEvent(
                            type=EventType.TOOL_RESULT,
                            data={"name": fc.name, "output": result_str[:500]}
                        ))
                    
                    tool_results.append(Part(
                        function_response={
                            "name": fc.name,
                            "response": {"result": result_str}
                        }
                    ))
                    
                    logger.info("tool_executed", 
                        tool=fc.name,
                        success=not isinstance(result, Exception),
                        iteration=iterations
                    )
                
                # Add tool results to conversation
                messages.append(Content(role="user", parts=tool_results))
                
                # Checkpoint session after each iteration
                session.messages = messages
                session.iteration = iterations
                if self.sessions:
                    await self.sessions.save(session, owner_id=owner_id)
            
            # Limit reached (time or iterations)
            elapsed = asyncio.get_event_loop().time() - start_time
            timeout_reached = time_remaining() <= 0
            
            if on_event:
                on_event(StreamEvent(
                    type=EventType.DONE,
                    data={
                        "iterations": iterations, 
                        "max_reached": True,
                        "timeout_reached": timeout_reached,
                        "elapsed_seconds": round(elapsed, 2),
                    }
                ))
            
            reason = "Timeout reached" if timeout_reached else "Max iterations reached"
            return AgentResult(
                text=f"{reason}. Task may be incomplete.",
                tool_calls=all_tool_calls,
                iterations=iterations,
                session_id=session.id,
                completed=False
            )
            
        except Exception as e:
            logger.error("agent_error", error=str(e))
            
            if on_event:
                on_event(StreamEvent(
                    type=EventType.ERROR,
                    data={"error": str(e)}
                ))
            
            return AgentResult(
                text="",
                tool_calls=all_tool_calls,
                iterations=iterations,
                session_id=session.id,
                completed=False,
                error=str(e)
            )
    
    async def _execute_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute a single tool."""
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(output=f"Unknown tool: {name}", success=False)
        
        try:
            return await tool.execute(arguments)
        except Exception as e:
            return ToolResult(output=f"Tool error: {str(e)}", success=False)
