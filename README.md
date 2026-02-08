# Gemini Code Doctor

**Gemini 3 Hackathon Entry** | [Live Demo](https://gemini-frontend-murex.vercel.app) | [API Docs](https://gemini-agent-hackathon-production.up.railway.app/docs)

> **"Your codebase's AI health checkup."**

A comprehensive AI-powered code health platform that combines **security scanning**, **verified bug detection**, and **evolution roadmap** into a single unified analysis.

## The Code Doctor Approach

Three-phase analysis powered by Gemini 3's 2-million token context window:

1. **🔒 Security Scan** - Pattern-based detection of secrets, credentials, and misconfigurations
2. **🐛 Code Analysis** - AI-powered bug detection with test verification (Vibe Engineering)
3. **📈 Evolution Advisor** - Strategic recommendations for codebase improvement

**Result:** A unified health report with overall score, actionable findings, and prioritized roadmap.

## Features

| Feature | Description |
|---------|-------------|
| **Security Pre-Scan** | 40+ secret patterns, entropy analysis, false positive reduction |
| **Verified Bug Detection** | AI identifies bugs, generates tests, confirms in E2B sandboxes |
| **AI Fix Proposals** | Proposed fixes for verified issues |
| **Evolution Roadmap** | Tech debt, architecture, quick wins, strategic initiatives |
| **Health Score** | Unified 0-100 score combining all analysis dimensions |
| **Real-Time Streaming** | SSE for live progress updates |
| **Full Observability** | Logs, metrics, diagnostics endpoints |

### What Gets Analyzed

| Category | Detection |
|----------|-----------|
| **Secrets** | API keys (OpenAI, Anthropic, Google, AWS, Stripe), database URLs, private keys, JWTs |
| **Bugs** | Logic errors, edge cases, type issues, error handling gaps |
| **Security** | Injection risks, auth issues, data exposure |
| **Evolution** | Technical debt, architecture patterns, testing gaps, dependencies |

## Quick Start

### Try the Demo

Visit [gemini-frontend-murex.vercel.app](https://gemini-frontend-murex.vercel.app)

1. Enter a GitHub repository URL
2. Toggle analysis types (Security, Code, Evolution)
3. Click "Run Code Doctor"
4. Watch the health checkup in real-time

### Run Locally

```bash
# Clone
git clone https://github.com/languageseed/gemini-agent-hackathon.git
cd gemini-agent-hackathon

# Setup
cp env.example .env
# Add GOOGLE_API_KEY and E2B_API_KEY to .env

# Install
pip install -r requirements.txt

# Run
./scripts/run_local.sh
# or: uvicorn src.main:app --reload
```

### Local Testing (Before Deploying!)

Run the analyzer locally on your own code before pushing to production:

```bash
# Analyze the src/ directory
python scripts/analyze_local.py

# Quick mode (no E2B verification - faster)
python scripts/analyze_local.py --quick

# Focus on security issues
python scripts/analyze_local.py --focus security

# Analyze a specific path
python scripts/analyze_local.py --path agent/

# Test API endpoints locally
python scripts/test_endpoints.py

# Test production endpoints
python scripts/test_endpoints.py --prod
```

This catches bugs before they hit Railway/Vercel!

## API Endpoints

### V5: Code Doctor (NEW)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v5/analyze/full` | POST | Full Code Doctor analysis |
| `/v5/analyze/full/stream` | POST | Code Doctor with SSE streaming |
| `/v5/analyze/security` | GET | Security scan only (fast) |
| `/v5/analyze/evolution` | POST | Evolution analysis only |

### V4: Verified Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v4/analyze/verified` | POST | Verified bug analysis |
| `/v4/analyze/verified/stream` | POST | With SSE streaming |
| `/v4/analyze/async` | POST | Submit async job |
| `/v4/jobs/{id}` | GET | Poll job status |

### Other

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with capabilities |
| `/diagnostics` | GET | Full system diagnostics |
| `/logs` | GET | Recent log entries |

### Example: Code Doctor

```bash
curl -X POST https://gemini-agent-hackathon-production.up.railway.app/v5/analyze/full \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "run_security_scan": true,
    "run_code_analysis": true,
    "run_evolution_analysis": true,
    "evolution_focus": "full",
    "max_issues_to_verify": 10
  }'
```

### Example: Python Client

```bash
pip install -e projects/gemini-analyst-client
export GEMINI_ANALYST_API_KEY="YOUR_API_KEY"

# CLI
gemini-analyst analyze https://github.com/owner/repo --focus security

# Python
from gemini_analyst import AnalystClient
client = AnalystClient(api_key=os.environ["GEMINI_ANALYST_API_KEY"])
report = client.analyze("https://github.com/owner/repo")
```

## Architecture

```
Frontend (Vercel)      Backend (Railway)              External Services
┌──────────────┐      ┌────────────────────────┐    ┌─────────────────┐
│  SvelteKit   │─────▶│  FastAPI v0.7.0        │───▶│  Gemini 3 Pro   │
│  Code Doctor │ SSE  │                        │    │  2M Context     │
│  Live UI     │      │  ┌──────────────────┐  │    ├─────────────────┤
└──────────────┘      │  │ Security Scanner │  │    │  E2B Sandbox    │
                      │  │ (Pattern-based)  │  │───▶│  Code Execution │
                      │  └──────────────────┘  │    ├─────────────────┤
                      │  ┌──────────────────┐  │    │  GitHub API     │
                      │  │ VerifiedAnalyzer │  │───▶│  Clone Repos    │
                      │  │ (Gemini-powered) │  │    └─────────────────┘
                      │  └──────────────────┘  │
                      │  ┌──────────────────┐  │
                      │  │ EvolutionAdvisor │  │
                      │  │ (Gemini-powered) │  │
                      │  └──────────────────┘  │
                      └────────────────────────┘
```

## Code Doctor Response

```json
{
  "repo_url": "https://github.com/owner/repo",
  "overall_health_score": 72,
  
  "security_findings": [...],
  "security_summary": {
    "total": 3,
    "critical": 1,
    "high": 1,
    "medium": 1
  },
  
  "code_issues": [...],
  "code_summary": {
    "total": 5,
    "verified": 2,
    "unverified": 3
  },
  
  "evolution_recommendations": [...],
  "evolution_summary": {
    "total": 8,
    "quick_wins": 3,
    "health_score": 65,
    "maturity_level": "growing"
  },
  
  "executive_summary": "..."
}
```

## Tech Stack

- **Frontend:** SvelteKit, TailwindCSS, Lucide Icons
- **Backend:** FastAPI, Python 3.11, structlog
- **AI:** Gemini 3 Pro Preview (gemini-3-pro-preview)
- **Execution:** E2B Code Interpreter
- **Hosting:** Vercel (frontend), Railway (backend)

## Hackathon Submission

### Track: Vibe Engineering

> "Agents that write AND verify code"

This project implements the complete Vibe Engineering loop:
- **Scan:** Pre-scan for secrets and misconfigurations
- **Analyze:** Gemini identifies bugs and issues
- **Verify:** Tests are executed to confirm findings
- **Fix:** AI generates fixes for verified bugs
- **Evolve:** Strategic roadmap for codebase improvement

### Judging Criteria

| Criteria | Weight | How We Score |
|----------|--------|--------------|
| Technical Execution | 40% | Deep Gemini 3 integration, 2M context, verification loop, evolution advisor |
| Innovation | 30% | Novel approach: unified health checkup combining security, analysis, evolution |
| Impact | 20% | Reduces false positives, provides actionable roadmap, saves developer time |
| Presentation | 10% | Live demo, architecture docs, video script |

### Submission Materials

- [x] [200-word description](docs/submission-description.md)
- [x] [Demo URL](https://gemini-frontend-murex.vercel.app)
- [x] [GitHub Repository](https://github.com/languageseed/gemini-agent-hackathon)
- [x] [Video Script](demo/video-script.md)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key |
| `E2B_API_KEY` | Yes | E2B sandbox key |
| `API_SECRET_KEY` | Yes | Backend auth key |
| `GEMINI_MODEL` | No | Default model (gemini-2.0-flash) |
| `GEMINI_REASONING_MODEL` | No | Reasoning model (gemini-3-pro-preview) |

## License

MIT - Gemini 3 Hackathon Project
