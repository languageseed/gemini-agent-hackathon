# Reference Code for Code Intelligence Platform

This folder contains reference implementations that can be integrated into the Gemini Code Intelligence platform.

---

## Folder Structure

```
_reference/
├── README.md                    # This file
├── COMPLIANCE.md                # Hackathon compliance guide
│
├── security/                    # Security scanning components
│   ├── secret_scanner.py        # 40+ credential patterns, entropy analysis
│   ├── container_scanner.py     # Container config security checks
│   ├── network_scanner.py       # Network exposure scanning
│   ├── compliance_mapper.py     # Finding → compliance framework mapping
│   ├── recommendations.py       # Security advice generation
│   └── models/
│       └── finding.py           # Unified finding data model
│
└── evolution/                   # Evolution & roadmap analysis
    ├── concept.md               # Philosophy document
    ├── review_prompt.txt        # Structured review prompt template
    └── scheduler.py             # Scheduled analysis service
```

---

## Component Summary

### Security Components

| File | Purpose | Lines | Key Features |
|------|---------|-------|--------------|
| `secret_scanner.py` | Detect exposed credentials | 618 | 40+ patterns (AWS, GitHub, Stripe, etc.), entropy analysis, false positive reduction |
| `container_scanner.py` | Container security checks | 379 | Trivy integration, privileged mode, root user, volume mounts, resource limits |
| `network_scanner.py` | Network exposure | ~300 | Port scanning, service detection |
| `compliance_mapper.py` | Map to frameworks | 382 | Pattern + semantic matching to 12 frameworks (NIST, CIS, HIPAA, etc.) |
| `recommendations.py` | Generate fix advice | 765 | Quick wins, hardening, network, monitoring, strategic |
| `models/finding.py` | Data model | 95 | Severity, category, evidence, remediation fields |

### Evolution Components

| File | Purpose | Key Features |
|------|---------|--------------|
| `concept.md` | Philosophy | Evolution advisor design, passive overnight review, scheduling |
| `review_prompt.txt` | Analysis prompt | 11-section structured review (alignment, drift, dependencies, roadmap) |
| `scheduler.py` | Scheduled runs | Flask service for scheduled analysis with web UI |

---

## Integration Plan

### Into Gemini Agent Backend

```
src/
├── agent/
│   ├── core.py                  # Existing agent loop
│   ├── verified_analysis.py     # Existing verification
│   │
│   ├── security/                # NEW: From security/
│   │   ├── __init__.py
│   │   ├── secret_scanner.py    # Pre-scan before Gemini
│   │   ├── container_scanner.py # Container checks
│   │   └── compliance.py        # Framework mapping (Gemini-based)
│   │
│   └── evolution/               # NEW: From evolution/
│       ├── __init__.py
│       └── advisor.py           # Gemini-powered evolution advice
│
└── main.py                      # Add /v5 endpoints
```

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/v4/analyze/verified` | Existing: verified code analysis |
| `/v5/analyze/security` | New: security-focused scan |
| `/v5/analyze/evolution` | New: roadmap/growth advice |
| `/v5/analyze/full` | New: complete code checkup |

---

## Key Patterns to Reuse

### 1. Finding Model

```python
class Finding:
    id: str
    title: str
    description: str
    severity: Severity  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: FindingCategory  # VULNERABILITY, SECRET_EXPOSURE, etc.
    file_path: Optional[str]
    line_number: Optional[int]
    recommendation: Optional[str]
    # + verification fields from existing Gemini Agent
```

### 2. Secret Detection Patterns

Pre-scan codebase for secrets before sending to Gemini:
- Saves tokens by catching obvious issues early
- Provides specific remediation guidance
- Reduces false positives

### 3. Evolution Prompt Structure

Structured review covering:
- Standards alignment
- Container/code/spec drift
- Dependencies (consumed and provided)
- Categorized findings (defects, code issues, roadmap)
- Persistence across restart/rebuild

---

## Compliance Notes

See `COMPLIANCE.md` for:
- What to keep as-is (regex patterns, Docker checks)
- What to replace with Gemini (Valet embeddings → Gemini semantic matching)
- Model substitutions (`gpt-5.2-codex` → `gemini-3-pro`)
