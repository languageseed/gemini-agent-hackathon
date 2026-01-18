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


# WebSearchTool removed - was a stub. Add real implementation when needed.


# ============================================================
# E2B CODE EXECUTION
# ============================================================

class VerificationResult:
    """Structured result from verification test execution."""
    
    def __init__(
        self, 
        status: str,  # passed, assertion_failed, import_error, runtime_error, timeout, env_error
        output: str,
        error_type: str = None,
        error_message: str = None,
    ):
        self.status = status
        self.output = output
        self.error_type = error_type
        self.error_message = error_message
    
    @property
    def is_verified(self) -> bool:
        """Only assertion failures count as verified bugs."""
        return self.status == "assertion_failed"
    
    @property  
    def is_unverified(self) -> bool:
        """Test passed = bug may be false positive."""
        return self.status == "passed"
    
    @property
    def is_error(self) -> bool:
        """Environmental/runtime errors - can't determine verification."""
        return self.status in ("import_error", "runtime_error", "timeout", "env_error")


def classify_test_result(success: bool, output: str, error_info: str = None) -> VerificationResult:
    """
    Classify test execution result into verification status.
    
    Critical: Only assertion failures = verified bug.
    Import/module errors = environment issue, NOT verified.
    """
    output_lower = output.lower() if output else ""
    error_lower = error_info.lower() if error_info else ""
    combined = output_lower + " " + error_lower
    
    if success:
        return VerificationResult("passed", output)
    
    # Check for assertion failures (these are VERIFIED bugs)
    assertion_patterns = [
        "assertionerror",
        "assert ",
        "assertion failed",
        "expected",
        "!=",
        "not equal",
    ]
    if any(p in combined for p in assertion_patterns):
        return VerificationResult(
            "assertion_failed", 
            output,
            error_type="AssertionError",
            error_message=error_info or output[:200],
        )
    
    # Check for import/module errors (NOT verified - environment issue)
    import_patterns = [
        "modulenotfounderror",
        "importerror", 
        "no module named",
        "cannot import",
        "module not found",
    ]
    if any(p in combined for p in import_patterns):
        return VerificationResult(
            "import_error",
            output,
            error_type="ImportError",
            error_message="Test requires modules not available in sandbox",
        )
    
    # Check for file/path errors
    file_patterns = [
        "filenotfounderror",
        "no such file",
        "path does not exist",
    ]
    if any(p in combined for p in file_patterns):
        return VerificationResult(
            "env_error",
            output,
            error_type="FileNotFoundError", 
            error_message="Test references files not in sandbox",
        )
    
    # Check for timeout
    if "timeout" in combined or "timed out" in combined:
        return VerificationResult(
            "timeout",
            output,
            error_type="TimeoutError",
            error_message="Test execution timed out",
        )
    
    # Generic runtime error
    return VerificationResult(
        "runtime_error",
        output,
        error_type="RuntimeError",
        error_message=error_info or output[:200],
    )


async def execute_code_in_sandbox(code: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Shared E2B code execution function.
    
    Returns:
        (success: bool, output: str)
    """
    api_key = os.environ.get("E2B_API_KEY")
    
    if api_key:
        try:
            from e2b_code_interpreter import Sandbox
            
            with Sandbox.create() as sandbox:
                execution = sandbox.run_code(code)
                
                output_parts = []
                if hasattr(execution, 'text') and execution.text:
                    output_parts.append(execution.text)
                elif hasattr(execution, 'logs'):
                    if execution.logs.stdout:
                        output_parts.append(execution.logs.stdout)
                    if execution.logs.stderr:
                        output_parts.append(f"STDERR: {execution.logs.stderr}")
                
                if hasattr(execution, 'error') and execution.error:
                    error_msg = str(execution.error)
                    if hasattr(execution.error, 'name'):
                        error_msg = f"{execution.error.name}: {execution.error.value}"
                    return False, f"ERROR: {error_msg}"
                
                return True, "\n".join(output_parts) or "Success (no output)"
                
        except Exception as e:
            logger.warning("e2b_fallback", error=str(e))
    
    # Local fallback (development only)
    if os.environ.get("ALLOW_LOCAL_EXEC", "false").lower() == "true":
        return await _execute_local(code, timeout)
    
    return False, "E2B_API_KEY not set and local execution disabled"


async def execute_verification_test(test_code: str, timeout: int = 30) -> VerificationResult:
    """
    Execute a verification test and classify the result.
    
    Returns a VerificationResult with proper classification:
    - assertion_failed: Bug is VERIFIED (test caught the issue)
    - passed: Bug is UNVERIFIED (may be false positive)
    - import_error/runtime_error/env_error: Cannot determine (environmental issue)
    """
    success, output = await execute_code_in_sandbox(test_code, timeout)
    
    # Extract error info if present
    error_info = None
    if "ERROR:" in output:
        error_info = output.split("ERROR:")[-1].strip()
    
    return classify_test_result(success, output, error_info)


async def _execute_local(code: str, timeout: int) -> tuple[bool, str]:
    """Local execution fallback - only for development."""
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
        return True, stdout + (f"\nSTDERR: {stderr}" if stderr else "") or "Success"
        
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {str(e)}"


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
    
    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        timeout = min(arguments.get("timeout", 30), 300)
        
        success, output = await execute_code_in_sandbox(code, timeout)
        return ToolResult(output=output, success=success)


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
