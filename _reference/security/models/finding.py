"""Finding data models."""
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, Any


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    """Categories of security findings."""
    # Core categories
    VULNERABILITY = "vulnerability"
    MISCONFIGURATION = "misconfiguration"
    SECRET_EXPOSURE = "secret_exposure"
    MALWARE = "malware"
    HARDENING = "hardening"
    NETWORK = "network"
    SUPPLY_CHAIN = "supply_chain"
    AI_SECURITY = "ai_security"
    COMPLIANCE = "compliance"
    
    # Authentication & Access Control (maps to NIST PR.AA, HIPAA, etc.)
    AUTHENTICATION = "authentication"
    ACCESS_CONTROL = "access_control"
    AUTHORIZATION = "authorization"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Data Protection (maps to encryption, secrets controls)
    ENCRYPTION = "encryption"
    DATA_PROTECTION = "data_protection"
    
    # Logging & Monitoring (maps to DE.CM, audit controls)
    LOGGING = "logging"
    MONITORING = "monitoring"
    AUDIT = "audit"
    
    # Infrastructure
    CONTAINER = "container"
    HOST = "host"
    CONFIGURATION = "configuration"
    
    # Operational
    BACKUP = "backup"
    PATCHING = "patching"
    INVENTORY = "inventory"


class Finding(BaseModel):
    """A security finding from an audit."""
    
    id: str = Field(..., description="Unique finding identifier")
    title: str = Field(..., description="Short title describing the finding")
    description: str = Field(..., description="Detailed description")
    severity: Severity = Field(..., description="Severity level")
    category: FindingCategory = Field(..., description="Finding category")
    
    # Location
    file_path: Optional[str] = Field(None, description="File path where found")
    line_number: Optional[int] = Field(None, description="Line number")
    endpoint: Optional[str] = Field(None, description="API endpoint if applicable")
    container_name: Optional[str] = Field(None, description="Container name if applicable")
    
    # Details
    cve_id: Optional[str] = Field(None, description="CVE identifier if applicable")
    cvss_score: Optional[float] = Field(None, description="CVSS score")
    cwe_id: Optional[str] = Field(None, description="CWE identifier")
    
    # Evidence
    evidence: Optional[str] = Field(None, description="Evidence/proof (redacted if sensitive)")
    raw_output: Optional[Any] = Field(None, description="Raw scanner output")
    
    # Remediation
    recommendation: Optional[str] = Field(None, description="Remediation recommendation")
    references: list[str] = Field(default_factory=list, description="Reference URLs")
    
    # Metadata
    scanner: str = Field(..., description="Scanner that found this")
    scan_time: Optional[datetime] = Field(None, description="When scan occurred")
    ai_verified: bool = Field(False, description="Whether AI verified this finding")
    ai_confidence: Optional[float] = Field(None, description="AI confidence score")
    false_positive: bool = Field(False, description="Marked as false positive")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
