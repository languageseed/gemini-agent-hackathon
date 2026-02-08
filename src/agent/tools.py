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
    """Perform mathematical calculations using safe AST-based evaluation."""
    
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
    
    # Whitelisted function names that can be called
    SAFE_FUNCTIONS = {
        "abs", "round", "min", "max", "pow",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "sqrt", "log", "log10", "log2", "exp", "expm1", "log1p",
        "floor", "ceil", "trunc", "fabs", "factorial", "gcd",
        "degrees", "radians", "hypot",
    }
    
    # Safe constants
    SAFE_CONSTANTS = {
        "pi": 3.141592653589793,
        "e": 2.718281828459045,
        "tau": 6.283185307179586,
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        expr = arguments.get("expression", "")
        try:
            result = self._safe_eval(expr)
            return ToolResult(output=f"{expr} = {result}")
        except Exception as e:
            return ToolResult(output=f"Error evaluating '{expr}': {e}", success=False)
    
    def _safe_eval(self, expr: str) -> float:
        """
        Safely evaluate a mathematical expression using AST parsing.
        
        This does NOT use eval(). Instead, it parses the expression into an AST
        and only allows safe operations (numbers, arithmetic, whitelisted functions).
        """
        import ast
        import math
        import operator
        
        # Allowed binary operators
        OPERATORS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        
        # Allowed unary operators
        UNARY_OPS = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
        
        # Allowed comparison operators
        COMPARE_OPS = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }
        
        def _eval_node(node):
            """Recursively evaluate an AST node."""
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            
            elif isinstance(node, ast.Constant):
                # Python 3.8+ uses ast.Constant for numbers
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant type: {type(node.value)}")
            
            elif isinstance(node, ast.Num):
                # Python 3.7 compatibility
                return node.n
            
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in OPERATORS:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                return OPERATORS[op_type](left, right)
            
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in UNARY_OPS:
                    raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
                operand = _eval_node(node.operand)
                return UNARY_OPS[op_type](operand)
            
            elif isinstance(node, ast.Compare):
                # Handle comparisons like 5 > 3
                left = _eval_node(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    op_type = type(op)
                    if op_type not in COMPARE_OPS:
                        raise ValueError(f"Unsupported comparison: {op_type.__name__}")
                    right = _eval_node(comparator)
                    if not COMPARE_OPS[op_type](left, right):
                        return False
                    left = right
                return True
            
            elif isinstance(node, ast.Call):
                # Function calls - only allow whitelisted functions
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls allowed")
                
                func_name = node.func.id
                if func_name not in self.SAFE_FUNCTIONS:
                    raise ValueError(f"Function not allowed: {func_name}")
                
                # Get the function from math module
                if not hasattr(math, func_name):
                    raise ValueError(f"Unknown function: {func_name}")
                
                func = getattr(math, func_name)
                args = [_eval_node(arg) for arg in node.args]
                return func(*args)
            
            elif isinstance(node, ast.Name):
                # Variable names - only allow constants
                name = node.id
                if name in self.SAFE_CONSTANTS:
                    return self.SAFE_CONSTANTS[name]
                raise ValueError(f"Unknown variable: {name}")
            
            elif isinstance(node, ast.Tuple):
                # Tuples for functions like min(1, 2, 3)
                return tuple(_eval_node(elt) for elt in node.elts)
            
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")
        
        # Parse the expression
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        # Evaluate the AST
        return _eval_node(tree)


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
    
    # Check for assertion failures or explicit test failures (these are VERIFIED bugs)
    assertion_patterns = [
        "assertionerror",
        "assert ",
        "assertion failed",
        "expected",
        "!=",
        "not equal",
        "systemexit: 1",     # sys.exit(1) = test failed = bug verified
        "systemexit(1)",
        "exit(1)",
        "fail:",             # Common test failure prefix
        "test failed",
        "bug exists",
        "vulnerability",
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
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default 30, max 300)
    
    Returns:
        (success: bool, output: str)
    """
    import asyncio
    import concurrent.futures
    
    api_key = os.environ.get("E2B_API_KEY")
    timeout = min(timeout, 300)  # Cap at 5 minutes
    
    if api_key:
        try:
            from e2b_code_interpreter import Sandbox
            
            def run_in_sandbox():
                """Synchronous function to run in executor."""
                with Sandbox.create() as sandbox:
                    return sandbox.run_code(code)
            
            # Run synchronous E2B code in thread pool with timeout
            loop = asyncio.get_event_loop()
            try:
                execution = await asyncio.wait_for(
                    loop.run_in_executor(None, run_in_sandbox),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("e2b_timeout", timeout=timeout)
                return False, f"ERROR: TimeoutError: Execution timed out after {timeout}s"
            
            # Process successful execution result
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
    
    # No local fallback - E2B sandbox is the only supported execution environment.
    # Local exec with raw exec() was removed for security (arbitrary code execution).
    return False, "E2B_API_KEY not set. Code execution requires E2B sandbox."


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
    
    # Max code size to prevent abuse (50KB is generous for any reasonable code)
    MAX_CODE_SIZE = 50_000
    
    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        timeout = min(arguments.get("timeout", 30), 300)
        
        # Enforce code size limit
        if len(code) > self.MAX_CODE_SIZE:
            return ToolResult(
                output=f"Code too large: {len(code)} chars (max {self.MAX_CODE_SIZE})",
                success=False
            )
        
        success, output = await execute_code_in_sandbox(code, timeout)
        return ToolResult(output=output, success=success)


# ============================================================
# FILE TOOLS - Disabled in production
# ============================================================
# File tools are disabled by default because code execution happens in
# E2B sandboxes (remote), while file tools would operate on the host
# filesystem. This mismatch creates a security risk: the agent could
# write arbitrary files to the host while thinking it's writing to
# the sandbox. Enable only for local development with ENABLE_FILE_TOOLS=true.

import tempfile

# Max file write size (prevent disk exhaustion)
MAX_FILE_WRITE_SIZE = 100_000  # 100KB

SANDBOX_DIR = os.path.join(tempfile.gettempdir(), "gemini-agent-sandbox")


def _file_tools_enabled() -> bool:
    """Check if file tools are explicitly enabled (opt-in, not opt-out)."""
    return os.environ.get("ENABLE_FILE_TOOLS", "false").lower() == "true"


def _get_sandboxed_path(path: str) -> str:
    """
    Ensure path is within sandbox directory.
    Prevents path traversal attacks.
    """
    # Create sandbox if it doesn't exist
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    
    # Normalize and join with sandbox
    safe_path = os.path.normpath(path).lstrip(os.sep)
    full_path = os.path.join(SANDBOX_DIR, safe_path)
    
    # Verify it's still within sandbox (prevent ../ attacks)
    if not os.path.abspath(full_path).startswith(os.path.abspath(SANDBOX_DIR)):
        raise ValueError("Path escapes sandbox")
    
    return full_path


class ReadFileTool(Tool):
    """Read file contents from sandbox. Disabled by default in production."""
    
    name = "read_file"
    description = "Read the contents of a file (sandboxed to temp directory)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative to sandbox)"
            }
        },
        "required": ["path"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        # File tools are disabled by default (opt-in only)
        if not _file_tools_enabled():
            return ToolResult(
                output="File tools are disabled. Code execution uses E2B sandbox.",
                success=False
            )
        
        path = arguments.get("path", "")
        try:
            safe_path = _get_sandboxed_path(path)
            with open(safe_path, 'r') as f:
                content = f.read()
            return ToolResult(output=content[:10000])  # Limit size
        except ValueError:
            return ToolResult(output="Security error: path not allowed", success=False)
        except FileNotFoundError:
            return ToolResult(output="File not found", success=False)
        except Exception:
            return ToolResult(output="Error reading file", success=False)


class WriteFileTool(Tool):
    """Write content to a file in sandbox. Disabled by default in production."""
    
    name = "write_file"
    description = "Write content to a file (sandboxed to temp directory)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative to sandbox)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        # File tools are disabled by default (opt-in only)
        if not _file_tools_enabled():
            return ToolResult(
                output="File tools are disabled. Code execution uses E2B sandbox.",
                success=False
            )
        
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        
        # Enforce write size limit to prevent disk exhaustion
        if len(content) > MAX_FILE_WRITE_SIZE:
            return ToolResult(
                output=f"Content too large: {len(content)} bytes (max {MAX_FILE_WRITE_SIZE})",
                success=False
            )
        
        try:
            safe_path = _get_sandboxed_path(path)
            # Create parent directories if needed
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            with open(safe_path, 'w') as f:
                f.write(content)
            return ToolResult(output=f"Successfully wrote {len(content)} bytes to sandbox/{path}")
        except ValueError:
            return ToolResult(output="Security error: path not allowed", success=False)
        except Exception:
            return ToolResult(output="Error writing file", success=False)


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
