# Demo Video Script (~3 minutes)

## Opening (0:00 - 0:15)

**[Show: Title card with project name]**

> "What if AI could not only find bugs in your code—but prove they exist?"
>
> "I'm going to show you Verified Codebase Analyst, powered by Gemini 3."

## The Problem (0:15 - 0:30)

**[Show: Screenshot of typical static analysis with many warnings]**

> "Traditional code analyzers produce endless false positives. Developers waste hours investigating issues that aren't real."
>
> "What if AI could verify its findings before reporting them?"

## The Solution (0:30 - 0:50)

**[Show: Architecture diagram]**

> "Verified Codebase Analyst takes a different approach:
> 1. Send the codebase to Gemini 3's large context window
> 2. Identify potential issues
> 3. Generate tests that FAIL if the bug exists
> 4. Execute those tests in sandboxes
> 5. Report verified issues with proposed fixes"

## Live Demo - Web UI (0:50 - 1:40)

**[Show: Frontend UI]**

> "Let me show you this in action."

**[Action: Paste a GitHub repo URL, select 'security' focus, enable 'Verify Findings']**

> "I'll analyze this project for security issues."

**[Action: Click 'Analyze & Verify']**

> "Watch the live progress—issues discovered, tests generated, verification running..."

**[Wait for completion, show results]**

> "Here's a verified security issue. The test proved it exists. And here's the AI-generated fix—one click to copy."

## Live Demo - CLI & API (1:40 - 2:20)

**[Show: Terminal]**

> "For CI/CD integration, we have a Python client."

**[Action: Run CLI command]**

```bash
gemini-analyst analyze https://github.com/owner/repo --focus bugs
```

> "Submit jobs asynchronously, poll for status, or set up webhooks for completion notifications."

**[Show: Python code example]**

```python
from gemini_analyst import AnalystClient
client = AnalystClient()
report = client.analyze("https://github.com/owner/repo")
print(f"Found {report.verified_count} verified bugs")
```

> "Full programmatic access—integrate verified analysis into your pipeline."

## Technical Highlights (2:20 - 2:45)

**[Show: /diagnostics endpoint]**

> "Under the hood:
> - Gemini 3 Pro Preview for reasoning
> - E2B sandboxes for safe test execution
> - Async job queue with concurrency limits
> - Full observability—metrics, logs, diagnostics"

## Closing (2:45 - 3:00)

**[Show: Title card with URLs]**

> "This is Vibe Engineering—AI that verifies its own work."
>
> "Verified Codebase Analyst. Try it at the link below."
>
> "Thank you."

---

## Recording Notes

- **Demo repos to use:** Pick 2-3 small repos with known issues
- **Fallback:** Have screenshots ready if API is slow
- **Terminal:** Keep visible to show real-time streaming
- **Timing:** Web demo ~50s, CLI demo ~40s, keeps under 3 min
