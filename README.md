# Verified Codebase Analyst

**Gemini 3 Hackathon Entry** | [Live Demo](https://gemini-frontend-murex.vercel.app) | [API Docs](https://gemini-agent-hackathon-production.up.railway.app/docs)

> **"AI that proves bugs exist before reporting them."**

An AI-powered code analyzer that uses Gemini 3's 2-million token context window to analyze codebases and **verify** findings through automated test generation.

## The Vibe Engineering Approach

Traditional static analysis produces false positives. We take a different approach:

1. **Analyze** - Load codebase into Gemini 3 (up to ~125K tokens)
2. **Extract** - Identify bugs, security issues, performance problems
3. **Verify** - Generate self-contained tests that FAIL if the bug exists
4. **Execute** - Run tests in E2B sandboxes (snippet-isolated)
5. **Propose** - Generate AI fix proposals for verified bugs

**Result:** Report issues with verification status and honest confidence levels.

## Features

| Feature | Description |
|---------|-------------|
| **Large Context Analysis** | No chunking or RAG - uses Gemini's 2M context window |
| **Automated Verification** | Tests generated and executed to confirm bugs (snippet-level) |
| **AI-Generated Fix Proposals** | Proposed fixes for verified issues (labeled as "proposed") |
| **Real-Time Streaming** | SSE for live progress updates |
| **Full Observability** | Logs, metrics, diagnostics endpoints |

### Transparency Notes
- Large repos are truncated to ~500K chars for reliability
- Verification runs snippet-isolated tests (not full repo integration)
- Fixes are proposals, not verified against the test suite

## Quick Start

### Try the Demo

Visit [gemini-frontend-murex.vercel.app](https://gemini-frontend-murex.vercel.app)

1. Enter a GitHub repository URL
2. Select focus (bugs, security, performance, etc.)
3. Enable "Verify Findings"
4. Watch issues get discovered and verified in real-time

### Run Locally

```bash
# Clone
git clone https://github.com/languageseed/gemini-agent-hackathon.git
cd gemini-agent-hackathon

# Setup
cp env.example .env
# Add GEMINI_API_KEY to .env

# Install
pip install -r requirements.txt

# Run
uvicorn src.main:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with version and capabilities |
| `/diagnostics` | GET | Full component health checks |
| `/logs` | GET | Recent activity buffer |
| `/v4/analyze/verified/stream` | POST | **Main endpoint** - Verified analysis with SSE |

### Example Request

```bash
curl -X POST https://gemini-agent-hackathon-production.up.railway.app/v4/analyze/verified/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "repo_url": "https://github.com/user/repo",
    "focus": "bugs",
    "max_issues_to_verify": 10
  }'
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed diagrams.

```
Frontend (Vercel)      Backend (Railway)         External Services
┌──────────────┐      ┌────────────────────┐    ┌─────────────────┐
│  SvelteKit   │─────▶│  FastAPI v0.5.0    │───▶│  Gemini 3 Pro   │
│  Live UI     │ SSE  │  VerifiedAnalyzer  │    │  2M Context     │
└──────────────┘      │  Observability     │    ├─────────────────┤
                      └────────────────────┘───▶│  E2B Sandbox    │
                                                │  Code Execution │
                                                ├─────────────────┤
                                                │  GitHub API     │
                                                │  Clone Repos    │
                                                └─────────────────┘
```

## Tech Stack

- **Frontend:** SvelteKit, TailwindCSS, Lucide Icons
- **Backend:** FastAPI, Python 3.11, structlog
- **AI:** Gemini 3 Pro Preview (gemini-3-pro-preview)
- **Execution:** E2B Code Interpreter
- **Hosting:** Vercel (frontend), Railway (backend)
- **Secrets:** Doppler

## Hackathon Submission

### Track: Vibe Engineering

> "Agents that write AND verify code"

This project implements the complete Vibe Engineering loop:
- **Write:** Gemini identifies issues and writes tests
- **Verify:** Tests are executed to confirm findings
- **Fix:** AI generates fixes for verified bugs

### Judging Criteria

| Criteria | Weight | How We Score |
|----------|--------|--------------|
| Technical Execution | 40% | Deep Gemini 3 integration, 2M context, verification loop |
| Innovation | 30% | Novel approach: AI that proves its own findings |
| Impact | 20% | Reduces false positives, saves developer time |
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
