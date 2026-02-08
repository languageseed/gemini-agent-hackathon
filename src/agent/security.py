"""
Security Scanning Module

Pre-scans codebase for security issues before Gemini analysis.
Uses pattern matching for secrets and configuration checks.

This is rule-based scanning (no AI) - compliant with Gemini hackathon.
"""

import re
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class SecretType(str, Enum):
    """Types of secrets that can be detected."""
    # AI/ML API Keys
    OPENAI_API_KEY = "openai_api_key"
    ANTHROPIC_API_KEY = "anthropic_api_key"
    GOOGLE_API_KEY = "google_api_key"
    
    # Cloud Providers
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    
    # Version Control
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    
    # Communication
    SLACK_TOKEN = "slack_token"
    DISCORD_TOKEN = "discord_token"
    
    # Payment
    STRIPE_KEY = "stripe_key"
    
    # Database
    DATABASE_URL = "database_url"
    
    # Cryptographic
    PRIVATE_KEY = "private_key"
    JWT_SECRET = "jwt_secret"
    
    # Generic
    PASSWORD = "password"
    API_KEY = "api_key"
    SECRET = "secret"
    HIGH_ENTROPY = "high_entropy"


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    """Categories of security findings."""
    SECRET_EXPOSURE = "secret_exposure"
    VULNERABILITY = "vulnerability"
    MISCONFIGURATION = "misconfiguration"
    HARDENING = "hardening"


@dataclass
class SecurityFinding:
    """A security finding from scanning."""
    id: str
    title: str
    description: str
    severity: Severity
    category: FindingCategory
    file_path: str
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    secret_type: Optional[SecretType] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "secret_type": self.secret_type.value if self.secret_type else None,
        }


@dataclass
class SecretPattern:
    """Definition of a secret pattern to detect."""
    secret_type: SecretType
    name: str
    pattern: str
    severity: Severity
    description: str
    recommendation: str
    entropy_threshold: Optional[float] = None


# Secret patterns library (subset of comprehensive patterns)
SECRET_PATTERNS: list[SecretPattern] = [
    # AI/ML API Keys
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        name="OpenAI API Key",
        pattern=r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}",
        severity=Severity.CRITICAL,
        description="OpenAI API key detected - provides access to GPT models",
        recommendation="Rotate key immediately. Use environment variables or secrets manager.",
    ),
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        name="OpenAI API Key (Project)",
        pattern=r"sk-proj-[A-Za-z0-9_-]{80,}",
        severity=Severity.CRITICAL,
        description="OpenAI project API key detected",
        recommendation="Rotate key immediately. Use environment variables.",
    ),
    SecretPattern(
        secret_type=SecretType.ANTHROPIC_API_KEY,
        name="Anthropic API Key",
        pattern=r"sk-ant-api\d{2}-[A-Za-z0-9_-]{80,}",
        severity=Severity.CRITICAL,
        description="Anthropic API key detected - provides access to Claude models",
        recommendation="Rotate key immediately. Store in environment variables.",
    ),
    SecretPattern(
        secret_type=SecretType.GOOGLE_API_KEY,
        name="Google API Key",
        pattern=r"AIza[A-Za-z0-9_-]{35}",
        severity=Severity.CRITICAL,
        description="Google API key detected (Gemini, Maps, etc.)",
        recommendation="Rotate key immediately. Use environment variables.",
    ),
    
    # AWS
    SecretPattern(
        secret_type=SecretType.AWS_ACCESS_KEY,
        name="AWS Access Key ID",
        pattern=r"(?:A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}",
        severity=Severity.CRITICAL,
        description="AWS Access Key ID detected",
        recommendation="Rotate AWS key. Use IAM roles or AWS Secrets Manager.",
    ),
    
    # GitHub
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub Personal Access Token",
        pattern=r"ghp_[A-Za-z0-9]{36}",
        severity=Severity.CRITICAL,
        description="GitHub Personal Access Token detected",
        recommendation="Revoke and regenerate token. Use fine-grained tokens with minimal scope.",
    ),
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub Fine-grained Token",
        pattern=r"github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}",
        severity=Severity.CRITICAL,
        description="GitHub Fine-grained Personal Access Token detected",
        recommendation="Revoke and regenerate. Use environment variables.",
    ),
    
    # Stripe
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        name="Stripe Secret Key (Live)",
        pattern=r"sk_live_[A-Za-z0-9]{24,}",
        severity=Severity.CRITICAL,
        description="Stripe Live Secret Key detected - can process real payments",
        recommendation="Rotate Stripe key immediately. Never expose live keys.",
    ),
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        name="Stripe Secret Key (Test)",
        pattern=r"sk_test_[A-Za-z0-9]{24,}",
        severity=Severity.MEDIUM,
        description="Stripe Test Secret Key detected",
        recommendation="Move to environment variables.",
    ),
    
    # Database URLs
    SecretPattern(
        secret_type=SecretType.DATABASE_URL,
        name="PostgreSQL Connection String",
        pattern=r"postgres(?:ql)?://[^:]+:[^@]+@[^/]+(?:/[^\s\"']+)?",
        severity=Severity.CRITICAL,
        description="PostgreSQL connection string with credentials detected",
        recommendation="Move credentials to environment variables or secrets manager.",
    ),
    SecretPattern(
        secret_type=SecretType.DATABASE_URL,
        name="MongoDB Connection String",
        pattern=r"mongodb(?:\+srv)?://[^:]+:[^@]+@[^\s\"']+",
        severity=Severity.CRITICAL,
        description="MongoDB connection string with credentials detected",
        recommendation="Move credentials to secrets manager.",
    ),
    
    # Private Keys
    SecretPattern(
        secret_type=SecretType.PRIVATE_KEY,
        name="RSA Private Key",
        pattern=r"-----BEGIN RSA PRIVATE KEY-----",
        severity=Severity.CRITICAL,
        description="RSA Private Key detected",
        recommendation="Remove from source control. Generate new key pair.",
    ),
    SecretPattern(
        secret_type=SecretType.PRIVATE_KEY,
        name="SSH Private Key",
        pattern=r"-----BEGIN OPENSSH PRIVATE KEY-----",
        severity=Severity.CRITICAL,
        description="OpenSSH Private Key detected",
        recommendation="Remove SSH key. Generate new keys.",
    ),
    
    # JWT
    SecretPattern(
        secret_type=SecretType.JWT_SECRET,
        name="JWT Token",
        pattern=r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        severity=Severity.HIGH,
        description="JSON Web Token detected",
        recommendation="Ensure JWTs are not hardcoded. Use dynamic generation.",
    ),
    
    # Generic patterns
    SecretPattern(
        secret_type=SecretType.PASSWORD,
        name="Hardcoded Password",
        pattern=r"(?i)(?:password|passwd|pwd|pass)['\"]?\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]",
        severity=Severity.HIGH,
        description="Hardcoded password detected",
        recommendation="Remove hardcoded password. Use environment variables.",
    ),
    SecretPattern(
        secret_type=SecretType.API_KEY,
        name="Generic API Key",
        pattern=r"(?i)(?:api[_\-]?key|apikey)['\"]?\s*[:=]\s*['\"]([A-Za-z0-9_-]{20,})['\"]",
        severity=Severity.MEDIUM,
        description="Generic API key detected",
        recommendation="Move API key to environment variables.",
        entropy_threshold=4.0,
    ),
]


def calculate_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not data:
        return 0.0
    
    entropy = 0.0
    for x in set(data):
        p_x = data.count(x) / len(data)
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    
    return entropy


# False positive indicators
PLACEHOLDER_INDICATORS = [
    "example", "sample", "test", "dummy", "fake", "placeholder",
    "xxx", "your_", "your-", "<your", "insert_", "changeme",
    "todo", "fixme", "replace_this", "mock", "stub",
]


def is_false_positive(value: str, line: str) -> bool:
    """Check if this is likely a false positive."""
    value_lower = value.lower()
    line_lower = line.lower()
    
    # Placeholder values
    if any(ind in value_lower or ind in line_lower for ind in PLACEHOLDER_INDICATORS):
        return True
    
    # All same character
    if len(set(value)) <= 2:
        return True
    
    # Comment lines with "example"
    if line.strip().startswith('#') or line.strip().startswith('//'):
        if 'example' in line_lower or 'sample' in line_lower:
            return True
    
    return False


def redact_secret(line: str, secret: str) -> str:
    """Redact the secret value in the line."""
    if len(secret) <= 8:
        redacted = '*' * len(secret)
    else:
        redacted = secret[:4] + '*' * (len(secret) - 8) + secret[-4:]
    
    return line.replace(secret, redacted)


class SecretScanner:
    """
    Scans code for exposed secrets and credentials.
    
    Uses pattern matching and entropy analysis.
    No AI required - pure rule-based scanning.
    """
    
    def __init__(
        self,
        patterns: list[SecretPattern] = None,
        entropy_threshold: float = 4.5,
    ):
        self.patterns = patterns or SECRET_PATTERNS
        self.entropy_threshold = entropy_threshold
        
        # Pre-compile patterns
        self._compiled = [
            (p, re.compile(p.pattern, re.MULTILINE | re.IGNORECASE))
            for p in self.patterns
        ]
    
    # Max content size per file to prevent ReDoS on large files
    MAX_SCAN_SIZE = 500_000  # 500KB per file
    
    def scan_content(self, content: str, file_path: str = "unknown") -> list[SecurityFinding]:
        """Scan text content for secrets."""
        findings: list[SecurityFinding] = []
        
        if not content:
            return findings
        
        # Truncate oversized files to prevent ReDoS
        if len(content) > self.MAX_SCAN_SIZE:
            logger.warning("scan_truncated", file=file_path, size=len(content))
            content = content[:self.MAX_SCAN_SIZE]
        
        lines = content.split('\n')
        
        for pattern_def, compiled in self._compiled:
            for match in compiled.finditer(content):
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                # Extract the secret value
                secret_value = match.group(1) if match.groups() else match.group(0)
                
                # Apply entropy check if specified
                if pattern_def.entropy_threshold:
                    entropy = calculate_entropy(secret_value)
                    if entropy < pattern_def.entropy_threshold:
                        continue
                
                # Check for false positives
                if is_false_positive(secret_value, line_content):
                    continue
                
                finding = SecurityFinding(
                    id=f"secret-{hashlib.md5(f'{file_path}:{line_num}:{pattern_def.name}'.encode()).hexdigest()[:12]}",
                    title=f"{pattern_def.name} Detected",
                    description=pattern_def.description,
                    severity=pattern_def.severity,
                    category=FindingCategory.SECRET_EXPOSURE,
                    file_path=file_path,
                    line_number=line_num,
                    evidence=redact_secret(line_content.strip(), secret_value),
                    recommendation=pattern_def.recommendation,
                    secret_type=pattern_def.secret_type,
                )
                findings.append(finding)
        
        # Also scan for high-entropy strings
        findings.extend(self._scan_high_entropy(content, file_path, lines))
        
        return self._deduplicate(findings)
    
    def _scan_high_entropy(
        self,
        content: str,
        file_path: str,
        lines: list[str],
    ) -> list[SecurityFinding]:
        """Scan for high-entropy strings that might be secrets."""
        findings: list[SecurityFinding] = []
        
        # Pattern for potential secrets
        high_entropy_pattern = re.compile(
            r'["\']([A-Za-z0-9+/=_-]{24,})["\']',
            re.MULTILINE
        )
        
        for match in high_entropy_pattern.finditer(content):
            value = match.group(1)
            entropy = calculate_entropy(value)
            
            if entropy >= self.entropy_threshold:
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                # Skip common false positives
                if is_false_positive(value, line_content):
                    continue
                
                # Check if surrounding context suggests a secret
                context_lower = line_content.lower()
                secret_indicators = ['key', 'token', 'secret', 'password', 'auth', 'credential', 'api']
                
                if any(ind in context_lower for ind in secret_indicators):
                    findings.append(SecurityFinding(
                        id=f"entropy-{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:12]}",
                        title="High Entropy String (Potential Secret)",
                        description=f"High-entropy string (entropy: {entropy:.2f}) in suspicious context",
                        severity=Severity.MEDIUM,
                        category=FindingCategory.SECRET_EXPOSURE,
                        file_path=file_path,
                        line_number=line_num,
                        evidence=redact_secret(line_content.strip(), value),
                        recommendation="Review this string to determine if it's a secret that should be externalized.",
                        secret_type=SecretType.HIGH_ENTROPY,
                    ))
        
        return findings
    
    def _deduplicate(self, findings: list[SecurityFinding]) -> list[SecurityFinding]:
        """Remove duplicate findings."""
        seen = set()
        unique = []
        
        for finding in findings:
            key = (finding.file_path, finding.line_number, finding.title)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        
        return unique


def _scan_codebase_sync(repo_content: str) -> list[SecurityFinding]:
    """
    Synchronous scan - runs in a thread to avoid blocking the event loop.
    CPU-bound regex operations should not run on the main async thread.
    """
    scanner = SecretScanner()
    all_findings = []
    
    # Parse the repo content into files
    # Format from clone_repo: "### path/to/file.py\n```python\ncontent\n```"
    file_pattern = re.compile(r'### ([^\n]+)\n```[^\n]*\n(.*?)```', re.DOTALL)
    
    for match in file_pattern.finditer(repo_content):
        file_path = match.group(1).strip()
        file_content = match.group(2)
        
        findings = scanner.scan_content(file_content, file_path)
        all_findings.extend(findings)
    
    return all_findings


def scan_codebase_for_secrets(repo_content: str) -> list[SecurityFinding]:
    """
    Scan repository content for exposed secrets.
    
    Runs synchronously (call from sync context or use async version).
    """
    return _scan_codebase_sync(repo_content)


async def scan_codebase_for_secrets_async(repo_content: str) -> list[SecurityFinding]:
    """
    Async-safe version: runs CPU-bound regex scanning in a thread pool
    to avoid blocking the event loop.
    """
    import asyncio
    return await asyncio.to_thread(_scan_codebase_sync, repo_content)


@dataclass
class SecurityScanResult:
    """Result of security scan."""
    findings: list[SecurityFinding]
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    secrets_found: int
    scan_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "total_findings": self.total_findings,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "secrets_found": self.secrets_found,
            "scan_time_seconds": self.scan_time_seconds,
        }
