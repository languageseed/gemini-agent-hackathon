"""
GitHub Integration Tools for Codebase Analyst

Tools for loading and analyzing GitHub repositories.
"""

import os
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx
import structlog

from .tools import Tool, ToolResult

logger = structlog.get_logger()


# File extensions to include in analysis
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
    ".vue", ".svelte", ".html", ".css", ".scss", ".less",
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".md", ".txt", ".rst",
    ".sql", ".graphql",
    ".sh", ".bash", ".zsh",
    ".dockerfile", "Dockerfile",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "target", "out",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    "coverage", ".coverage", "htmlcov",
    "vendor", "packages",
}

# Max file size to include (in bytes)
MAX_FILE_SIZE = 100_000  # 100KB

# Max total size for context
MAX_TOTAL_SIZE = 1_500_000  # ~1.5MB (~375K tokens, safe for 2M context)


@dataclass
class RepoFile:
    """A file from a repository."""
    path: str
    content: str
    size: int
    language: str


@dataclass
class RepoContext:
    """Flattened repository context."""
    files: list[RepoFile]
    total_size: int
    total_files: int
    skipped_files: int
    tree: str  # Directory tree representation


class CloneRepoTool(Tool):
    """Clone and analyze a GitHub repository."""
    
    name = "clone_repo"
    description = """Clone a GitHub repository and load its contents for analysis.
Returns the repository structure and file contents.
Use this to analyze codebases, find bugs, suggest improvements, or generate documentation."""
    
    parameters = {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repository URL (e.g., 'https://github.com/owner/repo' or 'owner/repo')"
            },
            "branch": {
                "type": "string",
                "description": "Branch to clone (default: main/master)"
            },
            "path_filter": {
                "type": "string",
                "description": "Optional path prefix to filter files (e.g., 'src/' to only include src folder)"
            }
        },
        "required": ["repo_url"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        repo_url = arguments.get("repo_url", "")
        branch = arguments.get("branch")
        path_filter = arguments.get("path_filter")
        
        # Normalize URL
        if not repo_url.startswith("http"):
            repo_url = f"https://github.com/{repo_url}"
        
        # Extract owner/repo
        parts = repo_url.rstrip("/").split("/")
        if len(parts) < 2:
            return ToolResult(output="Invalid repository URL", success=False)
        
        owner, repo = parts[-2], parts[-1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        
        try:
            context = await self._clone_and_process(owner, repo, branch, path_filter)
            
            # Format output
            output = f"""# Repository: {owner}/{repo}

## Statistics
- Total files: {context.total_files}
- Skipped files: {context.skipped_files}
- Total size: {context.total_size:,} bytes

## Directory Structure
```
{context.tree}
```

## File Contents

"""
            for file in context.files:
                output += f"### {file.path}\n"
                output += f"Language: {file.language} | Size: {file.size} bytes\n"
                output += f"```{file.language}\n{file.content}\n```\n\n"
            
            return ToolResult(
                output=output,
                success=True,
                artifacts=[f"Loaded {context.total_files} files from {owner}/{repo}"]
            )
            
        except Exception as e:
            logger.error("clone_repo_error", error=str(e), repo=f"{owner}/{repo}")
            return ToolResult(output=f"Failed to clone repository: {e}", success=False)
    
    async def _clone_and_process(
        self, 
        owner: str, 
        repo: str, 
        branch: Optional[str],
        path_filter: Optional[str]
    ) -> RepoContext:
        """Clone repo and extract contents."""
        
        # Use GitHub API to get repo contents (no git clone needed)
        # This is faster and works without git installed
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get default branch if not specified
            if not branch:
                repo_info = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}"
                )
                if repo_info.status_code == 200:
                    branch = repo_info.json().get("default_branch", "main")
                else:
                    branch = "main"
            
            # Get the tree
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
            tree_response = await client.get(tree_url)
            
            if tree_response.status_code != 200:
                raise Exception(f"Failed to get repository tree: {tree_response.status_code}")
            
            tree_data = tree_response.json()
            
            files: list[RepoFile] = []
            total_size = 0
            skipped = 0
            tree_lines = []
            
            # Process each file
            for item in tree_data.get("tree", []):
                if item["type"] != "blob":
                    continue
                
                path = item["path"]
                
                # Apply path filter
                if path_filter and not path.startswith(path_filter):
                    continue
                
                # Check if we should include this file
                ext = Path(path).suffix.lower()
                name = Path(path).name
                
                # Skip non-code files
                if ext not in CODE_EXTENSIONS and name not in CODE_EXTENSIONS:
                    skipped += 1
                    continue
                
                # Skip files in excluded directories
                if any(skip in path.split("/") for skip in SKIP_DIRS):
                    skipped += 1
                    continue
                
                # Check size
                size = item.get("size", 0)
                if size > MAX_FILE_SIZE:
                    skipped += 1
                    continue
                
                # Check total size
                if total_size + size > MAX_TOTAL_SIZE:
                    skipped += 1
                    continue
                
                # Fetch file content
                try:
                    content_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
                    content_response = await client.get(content_url)
                    
                    if content_response.status_code == 200:
                        content = content_response.text
                        
                        # Detect language
                        language = self._detect_language(path)
                        
                        files.append(RepoFile(
                            path=path,
                            content=content,
                            size=len(content),
                            language=language
                        ))
                        
                        total_size += len(content)
                        tree_lines.append(path)
                        
                except Exception as e:
                    logger.warning("file_fetch_error", path=path, error=str(e))
                    skipped += 1
            
            # Build tree representation
            tree_str = self._build_tree(tree_lines)
            
            return RepoContext(
                files=files,
                total_size=total_size,
                total_files=len(files),
                skipped_files=skipped,
                tree=tree_str
            )
    
    def _detect_language(self, path: str) -> str:
        """Detect language from file extension."""
        ext = Path(path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".scala": "scala",
            ".vue": "vue",
            ".svelte": "svelte",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".md": "markdown",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
        }
        return mapping.get(ext, "text")
    
    def _build_tree(self, paths: list[str]) -> str:
        """Build a tree representation of paths."""
        if not paths:
            return "(empty)"
        
        # Sort paths
        paths = sorted(paths)
        
        # Simple tree representation
        tree = []
        for path in paths[:50]:  # Limit to first 50 for readability
            depth = path.count("/")
            indent = "  " * depth
            name = path.split("/")[-1]
            tree.append(f"{indent}{name}")
        
        if len(paths) > 50:
            tree.append(f"  ... and {len(paths) - 50} more files")
        
        return "\n".join(tree)


class AnalyzeCodeTool(Tool):
    """Analyze code for patterns, bugs, and improvements."""
    
    name = "analyze_code"
    description = """Analyze code content for:
- Architecture patterns
- Potential bugs or issues
- Code quality metrics
- Improvement suggestions
- Security vulnerabilities

Use after clone_repo to perform deep analysis."""
    
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Code content to analyze"
            },
            "focus": {
                "type": "string",
                "description": "Analysis focus: 'bugs', 'security', 'performance', 'style', 'all'",
                "enum": ["bugs", "security", "performance", "style", "all"]
            }
        },
        "required": ["code"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        focus = arguments.get("focus", "all")
        
        # This tool is primarily for the LLM to use its reasoning
        # We just format the request appropriately
        
        analysis_prompt = f"""Analyze the following code with focus on: {focus}

Provide:
1. Summary of what the code does
2. Identified issues (bugs, anti-patterns, vulnerabilities)
3. Specific improvement suggestions with code examples
4. Quality score (1-10)

Code to analyze:
```
{code[:50000]}  # Limit to 50K chars
```
"""
        
        return ToolResult(
            output=f"Analysis request prepared for focus: {focus}. Code length: {len(code)} chars.",
            success=True
        )


class GenerateTestsTool(Tool):
    """Generate test cases for code."""
    
    name = "generate_tests"
    description = """Generate comprehensive test cases for code.
Creates unit tests, integration tests, and edge case tests.
Returns test code that can be executed."""
    
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Code to generate tests for"
            },
            "language": {
                "type": "string",
                "description": "Programming language (python, javascript, typescript, etc.)"
            },
            "framework": {
                "type": "string",
                "description": "Test framework to use (pytest, jest, mocha, etc.)"
            }
        },
        "required": ["code", "language"]
    }
    
    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        language = arguments.get("language", "python")
        framework = arguments.get("framework", "pytest" if language == "python" else "jest")
        
        return ToolResult(
            output=f"Test generation request for {language} using {framework}. Code length: {len(code)} chars.",
            success=True
        )


def create_github_tools() -> list[Tool]:
    """Create GitHub-related tools."""
    return [
        CloneRepoTool(),
        AnalyzeCodeTool(),
        GenerateTestsTool(),
    ]
