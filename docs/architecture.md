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
│                           BACKEND API (v0.5.0)                              │
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
   Input Repo URL ──────────▶ Fetch entire ──────────▶ Send to Gemini 3
   Select Focus                codebase                 with 2M context
   Enable Verify                                        window
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
- Sends entire codebase to Gemini (leverages 2M token context)
- Extracts structured issues with severity and category
- Generates Python tests to verify each issue
- Executes tests in E2B sandbox
- Generates fixes for verified bugs

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

## Hackathon Track: Vibe Engineering

This project implements the "Vibe Engineering" approach:

> **"Agents that write AND verify code"**

1. **Write**: Gemini analyzes code and identifies issues
2. **Verify**: Tests are generated and executed to confirm bugs
3. **Fix**: AI generates verified fixes for confirmed issues

This goes beyond simple code analysis by proving issues exist through automated testing.
