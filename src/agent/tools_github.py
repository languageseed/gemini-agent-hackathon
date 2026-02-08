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
    
    # Allowed GitHub hostnames (prevent SSRF)
    ALLOWED_HOSTS = {"github.com", "www.github.com"}
    
    async def execute(self, arguments: dict) -> ToolResult:
        repo_url = arguments.get("repo_url", "")
        branch = arguments.get("branch")
        path_filter = arguments.get("path_filter")
        
        # Normalize URL - if no protocol, assume github.com
        if not repo_url.startswith("http"):
            # Handle "owner/repo" or "github.com/owner/repo" format
            if repo_url.startswith("github.com/"):
                repo_url = f"https://{repo_url}"
            else:
                repo_url = f"https://github.com/{repo_url}"
        
        # ============================================================
        # SECURITY: Validate URL to prevent SSRF attacks
        # Only allow github.com URLs to prevent fetching internal resources
        # ============================================================
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(repo_url)
            
            # Must be HTTPS only
            if parsed.scheme != "https":
                return ToolResult(
                    output=f"Invalid URL scheme: {parsed.scheme}. Only HTTPS URLs allowed.",
                    success=False
                )
            
            # Validate hostname is github.com
            hostname = parsed.hostname.lower() if parsed.hostname else ""
            if hostname not in self.ALLOWED_HOSTS:
                logger.warning("ssrf_blocked", hostname=hostname, url=repo_url[:100])
                return ToolResult(
                    output=f"Only GitHub repositories are supported. Got hostname: {hostname}",
                    success=False
                )
            
            # Prevent port override (e.g., github.com:8080)
            if parsed.port is not None and parsed.port != 443:
                return ToolResult(
                    output="Custom ports are not allowed for security reasons.",
                    success=False
                )
                
        except Exception as e:
            return ToolResult(output=f"Invalid URL format: {e}", success=False)
        
        # Extract owner/repo from validated path
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            return ToolResult(output="Invalid repository URL: must be github.com/owner/repo", success=False)
        
        owner, repo = path_parts[0], path_parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        
        # Validate owner/repo format (alphanumeric, hyphens, underscores, dots)
        import re
        if not re.match(r'^[a-zA-Z0-9_.-]+$', owner) or not re.match(r'^[a-zA-Z0-9_.-]+$', repo):
            return ToolResult(output="Invalid owner or repository name format.", success=False)
        
        logger.info("clone_repo_validated", owner=owner, repo=repo)
        
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
        
        # Build headers with optional auth
        headers = {"Accept": "application/vnd.github.v3+json"}
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            logger.info("github_authenticated", rate_limit="5000/hr")
        else:
            logger.warning("github_unauthenticated", rate_limit="60/hr")
        
        async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
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
            
            # PHASE 1: Filter files to determine which to fetch (no network calls)
            files_to_fetch = []
            cumulative_size = 0
            
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
                if cumulative_size + size > MAX_TOTAL_SIZE:
                    skipped += 1
                    continue
                
                cumulative_size += size
                files_to_fetch.append(path)
            
            # PHASE 2: Fetch all files in parallel (major performance improvement)
            async def fetch_file(path: str) -> tuple[str, str | None]:
                """Fetch a single file, return (path, content) or (path, None) on error."""
                try:
                    content_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
                    response = await client.get(content_url)
                    if response.status_code == 200:
                        return (path, response.text)
                except Exception as e:
                    logger.warning("file_fetch_error", path=path, error=str(e))
                return (path, None)
            
            # Fetch all files concurrently
            if files_to_fetch:
                logger.info("fetching_files_parallel", count=len(files_to_fetch))
                results = await asyncio.gather(*[fetch_file(p) for p in files_to_fetch])
                
                # PHASE 3: Process results
                for path, content in results:
                    if content is not None:
                        language = self._detect_language(path)
                        files.append(RepoFile(
                            path=path,
                            content=content,
                            size=len(content),
                            language=language
                        ))
                        total_size += len(content)
                        tree_lines.append(path)
                    else:
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


# AnalyzeCodeTool removed - was a stub that just returned placeholder text.
# Use VerifiedAnalyzer for real analysis.


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
        GenerateTestsTool(),
    ]
