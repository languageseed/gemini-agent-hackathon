# Verified Codebase Analyst - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                         (SvelteKit on Vercel)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐ │
│  │ Repo Input  │  │ Live Progress   │  │    Analysis Results            │ │
│  │ Focus Select│  │ - Issues Found  │  │    - Verified Issues           │ │
│  │ Verify Toggle│ │ - Verification  │  │    - AI-Generated Fixes        │ │
│  │             │  │ - Fix Generation│  │    - Test Code & Output        │ │
│  └─────────────┘  └─────────────────┘  └─────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ SSE Streaming
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND API (v0.7.2)                              │
│                          (FastAPI on Railway)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     VerifiedAnalyzer Engine                          │   │
│  │                                                                      │   │
│  │  Phase 1: CLONE       Phase 2: ANALYZE      Phase 3: EXTRACT        │   │
│  │  ┌─────────────┐     ┌─────────────────┐   ┌──────────────────┐     │   │
│  │  │ GitHub API  │────▶│ Gemini 3 Pro    │──▶│ Structured       │     │   │
│  │  │ Clone Repo  │     │ ~2M Context     │   │ Issue Parsing    │     │   │
│  │  └─────────────┘     └─────────────────┘   └──────────────────┘     │   │
│  │         │                    │                      │               │   │
│  │         ▼                    ▼                      ▼               │   │
│  │  Phase 4: VERIFY      Phase 5: FIX          Phase 6: REPORT         │   │
│  │  ┌─────────────┐     ┌─────────────────┐   ┌──────────────────┐     │   │
│  │  │ Generate    │     │ Generate Fix    │   │ Summary with     │     │   │
│  │  │ Test Code   │────▶│ for Verified    │──▶│ Verified/Fixed   │     │   │
│  │  │ Execute in  │     │ Issues          │   │ Issues           │     │   │
│  │  │ E2B Sandbox │     └─────────────────┘   └──────────────────┘     │   │
│  │  └─────────────┘                                                    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐     │
│  │  Observability  │  │  Session Store  │  │   Security              │     │
│  │  - /diagnostics │  │  - Redis/Memory │  │   - API Key Auth        │     │
│  │  - /logs        │  │  - Persistence  │  │   - Pre-commit Hooks    │     │
│  │  - Metrics      │  │                 │  │   - Doppler Secrets     │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ API Calls
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐     │
│  │  Gemini 3 API   │  │  E2B Sandbox    │  │   GitHub API            │     │
│  │                 │  │                 │  │                         │     │
│  │  - Pro Preview  │  │  - Code Exec    │  │   - Clone Repos         │     │
│  │  - 2M Context   │  │  - Isolation    │  │   - Read Files          │     │
│  │  - Reasoning    │  │  - Python 3.11  │  │   - Public/Private      │     │
│  │                 │  │                 │  │                         │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. USER                     2. CLONE                    3. ANALYZE
   Input Repo URL ──────────▶ Fetch codebase ─────────▶ Send to Gemini 3
   Select Focus                (truncated to            (using ~125K tokens
   Enable Verify               ~500K chars)             of 2M available)
        │                                                    │
        │                                                    ▼
        │                     4. EXTRACT                5. VERIFY
        │                     Parse issues ◀──────────── Stream issues
        │                     from response               as found
        │                          │                         │
        │                          ▼                         │
        │                     For each issue:               │
        │                     ┌────────────┐                │
        │                     │ Generate   │                │
        │                     │ Python Test│                │
        │                     └─────┬──────┘                │
        │                           │                       │
        │                           ▼                       │
        │                     ┌────────────┐                │
        │                     │ Execute in │                │
        │                     │ E2B Sandbox│                │
        │                     └─────┬──────┘                │
        │                           │                       │
        │                           ▼                       │
        │                     Test FAILS? ────────────────▶ VERIFIED ✓
        │                     Test PASSES? ───────────────▶ UNVERIFIED ?
        │                           │                       │
        │                           ▼                       │
        │                     6. FIX GENERATION             │
        │                     For verified bugs:            │
        │                     ┌────────────┐                │
        │                     │ Generate   │                │
        │                     │ AI Fix     │                │
        │                     └─────┬──────┘                │
        │                           │                       │
        ◀───────────────────────────┴───────────────────────┘
   
   7. DISPLAY RESULTS
   - Issues with severity
   - Verification status
   - Test code & output
   - AI-generated fixes
```

## Key Components

### VerifiedAnalyzer (`verified_analysis.py`)
The core engine that orchestrates the analysis pipeline:
- Sends codebase to Gemini (truncated to ~500K chars / ~125K tokens for reliability)
- Extracts structured issues with severity and category
- Generates self-contained Python tests to verify each issue
- Executes tests in E2B sandbox (snippet-isolated, not full repo)
- Proposes fixes for verified bugs (fixes are AI-generated proposals, not verified)

### StreamEvent System (`stream.py`)
Real-time progress reporting via Server-Sent Events:
- `thinking` - Phase updates
- `tool_start` / `tool_result` - Tool execution
- `issue_found` - New issue discovered
- `verify_issue` - Verification result
- `generate_fix` - Fix generation result
- `done` - Analysis complete

### Observability (`main.py`)
Built-in monitoring and debugging:
- `/diagnostics` - Component health checks
- `/logs` - Recent activity buffer
- Request tracing with IDs
- Token usage tracking

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | SvelteKit, TailwindCSS, Lucide Icons |
| Backend | FastAPI, Python 3.11, structlog |
| AI | Gemini 3 Pro Preview (2M context) |
| Execution | E2B Code Interpreter Sandbox |
| Hosting | Vercel (frontend), Railway (backend) |
| Secrets | Doppler |
| Client | Python package with CLI (`gemini-analyst`) |

## API Features (v0.7.2)

### Async Job System
- `POST /v4/analyze/async` - Submit job, returns immediately
- `GET /v4/jobs/{id}` - Poll for status and results
- Webhook callbacks when job completes
- Concurrency limits (max 3 concurrent jobs)

### Python Client
```bash
pip install -e projects/gemini-analyst-client
gemini-analyst analyze https://github.com/owner/repo --async --wait
```

### Observability
- `/diagnostics/quick` - Metrics, uptime, error counts
- `/logs` - Recent activity buffer
- Request tracing with IDs

## Hackathon Track: Vibe Engineering

This project implements the "Vibe Engineering" approach:

> **"Agents that write AND verify code"**

1. **Analyze**: Gemini analyzes code and identifies potential issues
2. **Verify**: Self-contained tests are generated and executed to confirm bugs
   - Tests run in isolated sandbox (snippet-level, not full repo integration)
   - Only assertion failures count as "verified" (import/runtime errors don't)
3. **Propose**: AI generates fix proposals for confirmed issues
   - Fixes are labeled "proposed" (not verified against repo)

### Limitations (Transparency)
- Codebase truncated to ~500K chars for reliability
- Verification is snippet-isolated (code embedded in test, not repo mounted)
- Fix proposals are not re-tested against the failing test

This goes beyond simple code analysis by attempting to prove issues exist through automated testing, while being transparent about the verification boundaries.
