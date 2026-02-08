"""
Comprehensive Secrets and Sensitive Data Scanner.

Detects API keys, passwords, tokens, PII, and other sensitive data.
"""
import re
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from datetime import datetime

from src.models.finding import Finding, Severity, FindingCategory


class SecretType(str, Enum):
    """Types of secrets that can be detected."""
    # AI/ML API Keys
    OPENAI_API_KEY = "openai_api_key"
    ANTHROPIC_API_KEY = "anthropic_api_key"
    HUGGINGFACE_TOKEN = "huggingface_token"
    GOOGLE_API_KEY = "google_api_key"
    
    # Cloud Providers
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    AZURE_KEY = "azure_key"
    GCP_KEY = "gcp_key"
    DIGITALOCEAN_TOKEN = "digitalocean_token"
    
    # Version Control
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    
    # Communication
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    DISCORD_TOKEN = "discord_token"
    DISCORD_WEBHOOK = "discord_webhook"
    
    # Payment
    STRIPE_KEY = "stripe_key"
    
    # Database
    DATABASE_URL = "database_url"
    MONGODB_URI = "mongodb_uri"
    REDIS_URL = "redis_url"
    
    # Cryptographic
    RSA_PRIVATE_KEY = "rsa_private_key"
    SSH_PRIVATE_KEY = "ssh_private_key"
    PGP_PRIVATE_KEY = "pgp_private_key"
    JWT_SECRET = "jwt_secret"
    
    # Auth
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    
    # PII
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    
    # Generic
    PASSWORD = "password"
    API_KEY = "api_key"
    SECRET = "secret"
    HIGH_ENTROPY = "high_entropy"


@dataclass
class SecretPattern:
    """Definition of a secret pattern to detect."""
    secret_type: SecretType
    name: str
    pattern: str
    severity: Severity
    description: str
    entropy_threshold: Optional[float] = None
    keywords: list[str] = field(default_factory=list)


# Comprehensive pattern library
SECRET_PATTERNS: list[SecretPattern] = [
    # =========================================================================
    # AI/ML API Keys
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        name="OpenAI API Key",
        pattern=r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}",
        severity=Severity.CRITICAL,
        description="OpenAI API key - provides access to GPT models",
    ),
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        name="OpenAI API Key (Project)",
        pattern=r"sk-proj-[A-Za-z0-9_-]{80,}",
        severity=Severity.CRITICAL,
        description="OpenAI project API key",
    ),
    SecretPattern(
        secret_type=SecretType.ANTHROPIC_API_KEY,
        name="Anthropic API Key",
        pattern=r"sk-ant-api\d{2}-[A-Za-z0-9_-]{80,}",
        severity=Severity.CRITICAL,
        description="Anthropic API key - provides access to Claude models",
    ),
    SecretPattern(
        secret_type=SecretType.HUGGINGFACE_TOKEN,
        name="Hugging Face Token",
        pattern=r"hf_[A-Za-z0-9]{34,}",
        severity=Severity.HIGH,
        description="Hugging Face API token",
    ),
    SecretPattern(
        secret_type=SecretType.GOOGLE_API_KEY,
        name="Google API Key",
        pattern=r"AIza[A-Za-z0-9_-]{35}",
        severity=Severity.CRITICAL,
        description="Google API key (Gemini, Maps, etc.)",
    ),
    
    # =========================================================================
    # AWS
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.AWS_ACCESS_KEY,
        name="AWS Access Key ID",
        pattern=r"(?:A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}",
        severity=Severity.CRITICAL,
        description="AWS Access Key ID",
    ),
    SecretPattern(
        secret_type=SecretType.AWS_SECRET_KEY,
        name="AWS Secret Access Key",
        pattern=r"(?i)aws[_\-\.]?secret[_\-\.]?(?:access[_\-\.]?)?key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        severity=Severity.CRITICAL,
        description="AWS Secret Access Key",
        entropy_threshold=4.5,
    ),
    
    # =========================================================================
    # GitHub
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub Personal Access Token",
        pattern=r"ghp_[A-Za-z0-9]{36}",
        severity=Severity.CRITICAL,
        description="GitHub Personal Access Token",
    ),
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub OAuth Token",
        pattern=r"gho_[A-Za-z0-9]{36}",
        severity=Severity.CRITICAL,
        description="GitHub OAuth Access Token",
    ),
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub App Token",
        pattern=r"ghs_[A-Za-z0-9]{36}",
        severity=Severity.CRITICAL,
        description="GitHub App Installation Token",
    ),
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        name="GitHub Fine-grained Token",
        pattern=r"github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}",
        severity=Severity.CRITICAL,
        description="GitHub Fine-grained Personal Access Token",
    ),
    
    # =========================================================================
    # GitLab
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.GITLAB_TOKEN,
        name="GitLab Personal Access Token",
        pattern=r"glpat-[A-Za-z0-9_-]{20}",
        severity=Severity.CRITICAL,
        description="GitLab Personal Access Token",
    ),
    
    # =========================================================================
    # Slack
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.SLACK_TOKEN,
        name="Slack Bot Token",
        pattern=r"xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
        severity=Severity.HIGH,
        description="Slack Bot Token",
    ),
    SecretPattern(
        secret_type=SecretType.SLACK_TOKEN,
        name="Slack User Token",
        pattern=r"xoxp-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
        severity=Severity.HIGH,
        description="Slack User Token",
    ),
    SecretPattern(
        secret_type=SecretType.SLACK_WEBHOOK,
        name="Slack Webhook URL",
        pattern=r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
        severity=Severity.MEDIUM,
        description="Slack Incoming Webhook URL",
    ),
    
    # =========================================================================
    # Discord
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.DISCORD_TOKEN,
        name="Discord Bot Token",
        pattern=r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}",
        severity=Severity.HIGH,
        description="Discord Bot Token",
    ),
    SecretPattern(
        secret_type=SecretType.DISCORD_WEBHOOK,
        name="Discord Webhook URL",
        pattern=r"https://discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+",
        severity=Severity.MEDIUM,
        description="Discord Webhook URL",
    ),
    
    # =========================================================================
    # Stripe
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        name="Stripe Secret Key (Live)",
        pattern=r"sk_live_[A-Za-z0-9]{24,}",
        severity=Severity.CRITICAL,
        description="Stripe Live Secret Key - can process real payments",
    ),
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        name="Stripe Secret Key (Test)",
        pattern=r"sk_test_[A-Za-z0-9]{24,}",
        severity=Severity.MEDIUM,
        description="Stripe Test Secret Key",
    ),
    
    # =========================================================================
    # Database URLs
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.DATABASE_URL,
        name="PostgreSQL Connection String",
        pattern=r"postgres(?:ql)?://[^:]+:[^@]+@[^/]+(?:/[^\s\"']+)?",
        severity=Severity.CRITICAL,
        description="PostgreSQL connection string with credentials",
    ),
    SecretPattern(
        secret_type=SecretType.DATABASE_URL,
        name="MySQL Connection String",
        pattern=r"mysql://[^:]+:[^@]+@[^/]+(?:/[^\s\"']+)?",
        severity=Severity.CRITICAL,
        description="MySQL connection string with credentials",
    ),
    SecretPattern(
        secret_type=SecretType.MONGODB_URI,
        name="MongoDB Connection String",
        pattern=r"mongodb(?:\+srv)?://[^:]+:[^@]+@[^\s\"']+",
        severity=Severity.CRITICAL,
        description="MongoDB connection string with credentials",
    ),
    SecretPattern(
        secret_type=SecretType.REDIS_URL,
        name="Redis Connection String",
        pattern=r"redis://[^:]*:[^@]+@[^\s\"']+",
        severity=Severity.HIGH,
        description="Redis connection string with password",
    ),
    
    # =========================================================================
    # Cryptographic Keys
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.RSA_PRIVATE_KEY,
        name="RSA Private Key",
        pattern=r"-----BEGIN RSA PRIVATE KEY-----",
        severity=Severity.CRITICAL,
        description="RSA Private Key",
    ),
    SecretPattern(
        secret_type=SecretType.SSH_PRIVATE_KEY,
        name="SSH Private Key (OpenSSH)",
        pattern=r"-----BEGIN OPENSSH PRIVATE KEY-----",
        severity=Severity.CRITICAL,
        description="OpenSSH Private Key",
    ),
    SecretPattern(
        secret_type=SecretType.SSH_PRIVATE_KEY,
        name="SSH Private Key (EC)",
        pattern=r"-----BEGIN EC PRIVATE KEY-----",
        severity=Severity.CRITICAL,
        description="EC Private Key",
    ),
    SecretPattern(
        secret_type=SecretType.PGP_PRIVATE_KEY,
        name="PGP Private Key",
        pattern=r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
        severity=Severity.CRITICAL,
        description="PGP/GPG Private Key",
    ),
    
    # =========================================================================
    # Auth Tokens
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.JWT_SECRET,
        name="JWT Token",
        pattern=r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        severity=Severity.HIGH,
        description="JSON Web Token",
    ),
    SecretPattern(
        secret_type=SecretType.BASIC_AUTH,
        name="Basic Auth Header",
        pattern=r"(?i)(?:basic|bearer)\s+[A-Za-z0-9+/=]{20,}",
        severity=Severity.HIGH,
        description="HTTP Basic/Bearer Authentication header",
    ),
    
    # =========================================================================
    # PII
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.SSN,
        name="US Social Security Number",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        severity=Severity.CRITICAL,
        description="US Social Security Number (SSN)",
        keywords=["ssn", "social_security"],
    ),
    SecretPattern(
        secret_type=SecretType.CREDIT_CARD,
        name="Credit Card Number",
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        severity=Severity.CRITICAL,
        description="Credit Card Number (Visa, Mastercard, Amex, Discover)",
    ),
    
    # =========================================================================
    # Generic
    # =========================================================================
    SecretPattern(
        secret_type=SecretType.PASSWORD,
        name="Hardcoded Password",
        pattern=r"(?i)(?:password|passwd|pwd|pass)['\"]?\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]",
        severity=Severity.HIGH,
        description="Hardcoded password in configuration",
    ),
    SecretPattern(
        secret_type=SecretType.API_KEY,
        name="Generic API Key",
        pattern=r"(?i)(?:api[_\-]?key|apikey)['\"]?\s*[:=]\s*['\"]([A-Za-z0-9_-]{20,})['\"]",
        severity=Severity.MEDIUM,
        description="Generic API key",
        entropy_threshold=4.0,
    ),
    SecretPattern(
        secret_type=SecretType.SECRET,
        name="Generic Secret",
        pattern=r"(?i)(?:secret|token)['\"]?\s*[:=]\s*['\"]([A-Za-z0-9+/=_-]{20,})['\"]",
        severity=Severity.MEDIUM,
        description="Generic secret or token",
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


def luhn_checksum(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13:
        return False
    
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    
    return checksum % 10 == 0


class SecretsScanner:
    """
    Comprehensive secrets and sensitive data scanner.
    
    Features:
    - Pattern matching for 40+ secret types
    - Entropy analysis for high-randomness strings
    - Context-aware detection
    - False positive reduction
    """
    
    # False positive indicators
    PLACEHOLDER_INDICATORS = [
        "example", "sample", "test", "dummy", "fake", "placeholder",
        "xxx", "your_", "your-", "<your", "insert_", "changeme",
        "todo", "fixme", "replace_this", "mock", "stub",
    ]
    
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
    
    def scan_text(self, content: str, source: str = "unknown") -> list[Finding]:
        """Scan text content for secrets."""
        findings: list[Finding] = []
        
        if not content:
            return findings
        
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
                if self._is_false_positive(secret_value, line_content):
                    continue
                
                # Validate credit cards with Luhn
                if pattern_def.secret_type == SecretType.CREDIT_CARD:
                    if not luhn_checksum(secret_value):
                        continue
                
                finding = Finding(
                    id=f"secret-{hashlib.md5(f'{source}:{line_num}:{pattern_def.name}'.encode()).hexdigest()[:12]}",
                    title=f"{pattern_def.name} Detected",
                    description=pattern_def.description,
                    severity=pattern_def.severity,
                    category=FindingCategory.SECRET_EXPOSURE,
                    file_path=source,
                    line_number=line_num,
                    evidence=self._redact_secret(line_content.strip(), secret_value),
                    scanner="secrets_scanner",
                    scan_time=datetime.utcnow(),
                    recommendation=self._get_recommendation(pattern_def.secret_type),
                )
                findings.append(finding)
        
        # Also scan for high-entropy strings
        findings.extend(self._scan_high_entropy(content, source, lines))
        
        return self._deduplicate(findings)
    
    def _scan_high_entropy(
        self,
        content: str,
        source: str,
        lines: list[str],
    ) -> list[Finding]:
        """Scan for high-entropy strings that might be secrets."""
        findings: list[Finding] = []
        
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
                
                # Skip if already caught by specific patterns
                if self._matches_known_pattern(value):
                    continue
                
                # Skip common false positives
                if self._is_false_positive(value, line_content):
                    continue
                
                # Check if surrounding context suggests a secret
                context_lower = line_content.lower()
                secret_indicators = ['key', 'token', 'secret', 'password', 'auth', 'credential', 'api']
                
                if any(ind in context_lower for ind in secret_indicators):
                    findings.append(Finding(
                        id=f"entropy-{hashlib.md5(f'{source}:{line_num}'.encode()).hexdigest()[:12]}",
                        title="High Entropy String (Potential Secret)",
                        description=f"High-entropy string (entropy: {entropy:.2f}) in suspicious context",
                        severity=Severity.MEDIUM,
                        category=FindingCategory.SECRET_EXPOSURE,
                        file_path=source,
                        line_number=line_num,
                        evidence=self._redact_secret(line_content.strip(), value),
                        scanner="entropy_scanner",
                        scan_time=datetime.utcnow(),
                        recommendation="Review this string to determine if it's a secret that should be externalized.",
                    ))
        
        return findings
    
    def _matches_known_pattern(self, value: str) -> bool:
        """Check if value matches a known specific pattern."""
        for _, compiled in self._compiled:
            if compiled.fullmatch(value):
                return True
        return False
    
    def _is_false_positive(self, value: str, line: str) -> bool:
        """Check if this is likely a false positive."""
        value_lower = value.lower()
        line_lower = line.lower()
        
        # Placeholder values
        if any(ind in value_lower or ind in line_lower for ind in self.PLACEHOLDER_INDICATORS):
            return True
        
        # All same character
        if len(set(value)) <= 2:
            return True
        
        # Common base64 encoded test strings
        common_base64 = ['dGVzdA==', 'ZXhhbXBsZQ==', 'cGFzc3dvcmQ=']
        if value in common_base64:
            return True
        
        # Comment lines with "example"
        if line.strip().startswith('#') or line.strip().startswith('//'):
            if 'example' in line_lower or 'sample' in line_lower:
                return True
        
        return False
    
    def _redact_secret(self, line: str, secret: str) -> str:
        """Redact the secret value in the line."""
        if len(secret) <= 8:
            redacted = '*' * len(secret)
        else:
            redacted = secret[:4] + '*' * (len(secret) - 8) + secret[-4:]
        
        return line.replace(secret, redacted)
    
    def _get_recommendation(self, secret_type: SecretType) -> str:
        """Get remediation recommendation for secret type."""
        recommendations = {
            SecretType.AWS_ACCESS_KEY: "Rotate AWS access key immediately. Use IAM roles or AWS Secrets Manager.",
            SecretType.AWS_SECRET_KEY: "Rotate AWS secret key. Never commit to source control.",
            SecretType.OPENAI_API_KEY: "Rotate OpenAI API key. Use environment variables or secrets manager.",
            SecretType.ANTHROPIC_API_KEY: "Rotate Anthropic API key. Store in environment variables.",
            SecretType.GITHUB_TOKEN: "Revoke and regenerate GitHub token. Use fine-grained tokens with minimal scope.",
            SecretType.DATABASE_URL: "Move database credentials to environment variables or secrets manager.",
            SecretType.RSA_PRIVATE_KEY: "Remove private key from source control. Generate new key pair.",
            SecretType.SSH_PRIVATE_KEY: "Remove SSH key. Generate new keys and update authorized_keys.",
            SecretType.STRIPE_KEY: "Rotate Stripe key immediately. Never expose live keys.",
            SecretType.SSN: "PII exposure - remove SSN immediately. Review data handling procedures.",
            SecretType.CREDIT_CARD: "PCI-DSS violation - remove credit card data immediately.",
            SecretType.PASSWORD: "Remove hardcoded password. Use environment variables or secrets manager.",
        }
        return recommendations.get(
            secret_type,
            "Remove secret from source code. Use environment variables or a secrets manager."
        )
    
    def _deduplicate(self, findings: list[Finding]) -> list[Finding]:
        """Remove duplicate findings."""
        seen = set()
        unique = []
        
        for finding in findings:
            key = (finding.file_path, finding.line_number, finding.title)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        
        return unique
