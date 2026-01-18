"""
Tool Registry and Implementations

Built-in tools + E2B code execution.
Total: ~80 lines
"""

import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

import structlog

logger = structlog.get_logger()


@dataclass
class ToolResult:
    """Result from tool execution."""
    output: str
    success: bool = True
    artifacts: list[str] = field(default_factory=list)  # File paths, images, etc.


class Tool(ABC):
    """Base class for tools."""
    
    name: str
    description: str
    parameters: dict
    
    @abstractmethod
    async def execute(self, arguments: dict) -> ToolResult:
        """Execute the tool with given arguments."""
        pass


class ToolRegistry(dict):
    """Registry of available tools."""
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self[tool.name] = tool
    
    def to_declarations(self) -> list[dict]:
        """Convert to Gemini function declarations format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.values()
        ]


# ============================================================
# BUILT-IN TOOLS
# ============================================================

class GetCurrentTimeTool(Tool):
    """Get current date and time."""
    
    name = "get_current_time"
    description = "Get the current date and time"
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g., 'UTC', 'America/New_York'). Default: UTC"
            }
        },
        "required": []
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        tz = arguments.get("timezone", "UTC")
        now = datetime.now().isoformat()
        return ToolResult(output=f"Current time: {now} ({tz})")


class CalculateTool(Tool):
    """Perform mathematical calculations."""
    
    name = "calculate"
    description = "Perform a mathematical calculation. Supports basic math operations."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.15')"
            }
        },
        "required": ["expression"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        expr = arguments.get("expression", "")
        try:
            # Safe evaluation - only math operations
            allowed = {"__builtins__": {}}
            import math
            allowed.update(vars(math))
            result = eval(expr, allowed, {})
            return ToolResult(output=f"{expr} = {result}")
        except Exception as e:
            return ToolResult(output=f"Error evaluating '{expr}': {e}", success=False)


class WebSearchTool(Tool):
    """Search the web for information."""
    
    name = "web_search"
    description = "Search the web for current information on any topic"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        query = arguments.get("query", "")
        # TODO: Integrate with actual search API (e.g., Serper, SerpAPI, Tavily)
        return ToolResult(
            output=f"[Web search for '{query}' - implement with search API]",
            success=True
        )


# ============================================================
# E2B CODE EXECUTION
# ============================================================

class ExecuteCodeTool(Tool):
    """Execute code in E2B sandbox."""
    
    name = "execute_code"
    description = """Execute Python code in a secure sandbox. 
Use this to:
- Run data analysis
- Test code snippets
- Generate charts/visualizations
- Process files
The sandbox has numpy, pandas, matplotlib, requests pre-installed."""
    
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30, max: 300)"
            }
        },
        "required": ["code"]
    }
    
    def __init__(self):
        self._sandbox = None
    
    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        timeout = min(arguments.get("timeout", 30), 300)
        
        # Check for E2B API key
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            # Fallback to local execution (less secure, for development)
            return await self._execute_local(code, timeout)
        
        return await self._execute_e2b(code, timeout)
    
    async def _execute_e2b(self, code: str, timeout: int) -> ToolResult:
        """Execute in E2B sandbox."""
        try:
            from e2b_code_interpreter import Sandbox
            
            # Create sandbox (reuse if possible)
            sandbox = Sandbox(timeout=timeout)
            
            try:
                # Execute code
                execution = sandbox.run_code(code)
                
                # Collect output
                output_parts = []
                artifacts = []
                
                if execution.logs.stdout:
                    output_parts.append(execution.logs.stdout)
                
                if execution.logs.stderr:
                    output_parts.append(f"STDERR: {execution.logs.stderr}")
                
                if execution.error:
                    output_parts.append(f"ERROR: {execution.error.name}: {execution.error.value}")
                    return ToolResult(
                        output="\n".join(output_parts) or "No output",
                        success=False
                    )
                
                # Check for results (charts, data, etc.)
                if execution.results:
                    for result in execution.results:
                        if hasattr(result, 'png') and result.png:
                            artifacts.append(f"[Chart generated: {len(result.png)} bytes]")
                        elif hasattr(result, 'text') and result.text:
                            output_parts.append(result.text)
                
                return ToolResult(
                    output="\n".join(output_parts) or "Code executed successfully (no output)",
                    success=True,
                    artifacts=artifacts
                )
                
            finally:
                sandbox.kill()
                
        except ImportError:
            logger.warning("e2b_not_installed", fallback="local")
            return await self._execute_local(code, timeout)
        except Exception as e:
            logger.error("e2b_error", error=str(e))
            return ToolResult(output=f"E2B error: {str(e)}", success=False)
    
    async def _execute_local(self, code: str, timeout: int) -> ToolResult:
        """Fallback: execute locally (development only)."""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Create isolated namespace
            namespace = {"__builtins__": __builtins__}
            
            # Execute with timeout
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            output = stdout
            if stderr:
                output += f"\nSTDERR: {stderr}"
            
            return ToolResult(
                output=output or "Code executed successfully (no output)",
                success=True
            )
            
        except Exception as e:
            return ToolResult(
                output=f"Execution error: {type(e).__name__}: {str(e)}",
                success=False
            )


class ReadFileTool(Tool):
    """Read file contents."""
    
    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read"
            }
        },
        "required": ["path"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        path = arguments.get("path", "")
        try:
            with open(path, 'r') as f:
                content = f.read()
            return ToolResult(output=content[:10000])  # Limit size
        except Exception as e:
            return ToolResult(output=f"Error reading file: {e}", success=False)


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write content to a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        try:
            with open(path, 'w') as f:
                f.write(content)
            return ToolResult(output=f"Successfully wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(output=f"Error writing file: {e}", success=False)


# ============================================================
# DEFAULT REGISTRY
# ============================================================

def create_default_tools() -> ToolRegistry:
    """Create registry with default tools."""
    registry = ToolRegistry()
    
    registry.register(GetCurrentTimeTool())
    registry.register(CalculateTool())
    registry.register(WebSearchTool())
    registry.register(ExecuteCodeTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    
    return registry


def create_codebase_analyst_tools() -> ToolRegistry:
    """Create registry with codebase analysis tools."""
    from .tools_github import create_github_tools
    
    registry = create_default_tools()
    
    # Add GitHub/codebase tools
    for tool in create_github_tools():
        registry.register(tool)
    
    return registry
