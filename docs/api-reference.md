# Gemini Analyst API Reference

Base URL: `https://gemini-agent-hackathon-production.up.railway.app`

## Authentication

All endpoints (except `/health`) require an API key via the `X-API-Key` header:

```bash
curl -H "X-API-Key: YOUR_KEY" https://api.example.com/v4/analyze/verified
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/v3/analyze` | POST | Basic analysis |
| `/v3/analyze/stream` | POST | Basic analysis with SSE streaming |
| `/v4/analyze/verified` | POST | Verified analysis (tests run) |
| `/v4/analyze/verified/stream` | POST | Verified analysis with streaming |
| `/v4/analyze/async` | POST | Submit async job |
| `/v4/jobs/{job_id}` | GET | Get job status |
| `/v4/jobs` | GET | List jobs |
| `/v4/jobs/{job_id}` | DELETE | Cancel job |

---

## Health Check

```
GET /health
```

No authentication required. Returns API status and capabilities.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.6.0",
  "secured": true,
  "models": {
    "default": "gemini-2.0-flash",
    "reasoning": "gemini-3-pro-preview"
  },
  "capabilities": ["code_execution", "github_integration", "verified_analysis"],
  "config": {
    "e2b_configured": true,
    "gemini_configured": true
  }
}
```

---

## Verified Analysis

```
POST /v4/analyze/verified
```

Analyzes a repository and runs tests to verify findings.

**Request:**
```json
{
  "repo_url": "https://github.com/owner/repo",
  "branch": "main",
  "focus": "full",
  "verify": true,
  "generate_fixes": true,
  "max_issues_to_verify": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_url` | string | required | GitHub repository URL |
| `branch` | string | "main" | Branch to analyze |
| `focus` | string | "full" | Focus area: full, bugs, security, performance, architecture |
| `verify` | bool | true | Run tests to verify findings |
| `generate_fixes` | bool | true | Generate fix suggestions |
| `max_issues_to_verify` | int | 10 | Max issues to verify with tests |

**Response:**
```json
{
  "repo": "owner/repo",
  "summary": "Found 5 issues in the codebase...",
  "issues": [
    {
      "title": "SQL Injection vulnerability",
      "severity": "critical",
      "category": "security",
      "file_path": "src/db.py",
      "line_number": 42,
      "description": "User input is directly interpolated...",
      "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
      "recommendation": "Use parameterized queries",
      "verification_status": "verified",
      "verification_method": "snippet",
      "test_code": "def test_sql_injection():\n    ...",
      "test_output": "AssertionError: Vulnerable to injection",
      "fix_status": "proposed",
      "fix_code": "query = \"SELECT * FROM users WHERE id = ?\"\ncursor.execute(query, (user_id,))",
      "fix_description": "Use parameterized query to prevent injection"
    }
  ],
  "stats": {
    "total": 5,
    "verified": 3,
    "unverified": 1,
    "errors": 1,
    "by_severity": {"critical": 1, "high": 2, "medium": 2},
    "by_category": {"security": 2, "bugs": 2, "performance": 1}
  }
}
```

**Issue Fields:**

| Field | Description |
|-------|-------------|
| `severity` | critical, high, medium, low, info |
| `verification_status` | verified (test failed = bug confirmed), unverified (test passed), error, pending, skipped |
| `fix_status` | none, proposed (AI suggested), verified, failed |

---

## Streaming Analysis

```
POST /v4/analyze/verified/stream
```

Same request as above, but returns Server-Sent Events (SSE) for real-time progress.

**Event Types:**

| Event | Description | Data |
|-------|-------------|------|
| `thinking` | Phase update | `{phase: "Analyzing security..."}` |
| `issue_found` | Issue discovered | `{title, severity, category}` |
| `verify_start` | Starting verification | `{issue_title}` |
| `verify_result` | Verification complete | `{issue_title, status, output}` |
| `fix_generated` | Fix proposed | `{issue_title, fix_description}` |
| `done` | Analysis complete | Full report object |
| `error` | Error occurred | `{error: "message"}` |
| `heartbeat` | Keep-alive | `{}` |

**Example:**
```javascript
const eventSource = new EventSource('/v4/analyze/verified/stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'thinking':
      console.log(`Phase: ${data.phase}`);
      break;
    case 'issue_found':
      console.log(`Found: ${data.title}`);
      break;
    case 'done':
      console.log(`Complete: ${data.issues.length} issues`);
      eventSource.close();
      break;
  }
};
```

---

## Async Jobs

### Submit Job

```
POST /v4/analyze/async
```

Submit analysis for background processing. Returns immediately.

**Request:**
```json
{
  "repo_url": "https://github.com/owner/repo",
  "focus": "security",
  "verify": true,
  "webhook_url": "https://your-server.com/callback"
}
```

**Response:**
```json
{
  "job_id": "abc12345",
  "status": "pending",
  "status_url": "/v4/jobs/abc12345",
  "estimated_seconds": 120
}
```

### Get Job Status

```
GET /v4/jobs/{job_id}
```

**Response (running):**
```json
{
  "job_id": "abc12345",
  "status": "running",
  "progress": 0.45,
  "current_phase": "Verifying issues",
  "result": null,
  "error": null
}
```

**Response (completed):**
```json
{
  "job_id": "abc12345",
  "status": "completed",
  "progress": 1.0,
  "current_phase": "Complete",
  "result": { ... full analysis report ... },
  "error": null
}
```

### List Jobs

```
GET /v4/jobs?limit=20
```

Returns list of recent jobs.

### Cancel Job

```
DELETE /v4/jobs/{job_id}
```

Cancels a pending or running job.

---

## Webhook Notifications

When you provide a `webhook_url`, the API will POST to that URL when the job completes:

**Webhook Payload:**
```json
{
  "job_id": "abc12345",
  "status": "completed",
  "result": { ... full analysis report ... },
  "error": null
}
```

For failed jobs:
```json
{
  "job_id": "abc12345",
  "status": "failed",
  "result": null,
  "error": "Failed to clone repository"
}
```

---

## Python Client

Install the official Python client:

```bash
pip install gemini-analyst
```

Usage:

```python
from gemini_analyst import AnalystClient

client = AnalystClient(api_key="your-key")

# Synchronous analysis
report = client.analyze("https://github.com/owner/repo")

# With streaming progress
def on_progress(event_type, data):
    print(f"{event_type}: {data}")

report = client.analyze_stream(url, on_progress=on_progress)

# Async job with webhook
job = client.submit_async(url, webhook_url="https://your-server.com/callback")

# Or poll for completion
report = client.wait_for_job(job.job_id, timeout=300)
```

See the [client documentation](../gemini-analyst-client/README.md) for full details.

---

## Rate Limits

| Tier | Requests/hour | Notes |
|------|---------------|-------|
| Demo | 10 | No API key |
| Standard | 100 | With API key |
| Async jobs | 20 concurrent | Per API key |

When rate limited, you'll receive:
```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 3600
}
```

---

## Error Responses

| Status | Description |
|--------|-------------|
| 401 | Missing or invalid API key |
| 403 | API key lacks permission |
| 404 | Resource not found (job ID) |
| 422 | Invalid request body |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

Error format:
```json
{
  "detail": "Error description here"
}
```
