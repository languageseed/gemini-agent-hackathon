# Gemini Code Doctor - Demo Video Script (~3 minutes)

## Opening (0:00 - 0:15)

**[Show: Title card "Gemini Code Doctor"]**

> "What if AI could not only find bugs in your code—but prove they exist?"
>
> "I'm going to show you Gemini Code Doctor, a Vibe Engineering agent powered by Gemini 3."

## The Problem (0:15 - 0:30)

**[Show: Screenshot of typical static analysis with many warnings]**

> "Traditional code analyzers produce endless false positives. Developers waste hours investigating issues that aren't real."
>
> "What if AI could verify its findings before reporting them?"

## The Solution (0:30 - 0:50)

**[Show: Architecture diagram]**

> "Gemini Code Doctor takes a three-phase approach:
> 1. **Security Scan** - Find secrets and vulnerabilities
> 2. **Verified Analysis** - Identify bugs, generate tests, prove they exist
> 3. **Evolution Advice** - Recommend architecture improvements
>
> All powered by Gemini 3's 2-million token context window."

## Live Demo - Self-Analysis (0:50 - 1:50)

**[Show: Frontend UI at gemini-frontend-murex.vercel.app]**

> "Let me show you something special—the agent analyzing its own codebase."

**[Action: Paste https://github.com/languageseed/gemini-agent-hackathon]**

> "I'm going to have Code Doctor analyze itself."

**[Action: Click 'Analyze & Verify']**

> "Watch the live progress—cloning, analyzing, discovering issues..."

**[Show progress: Issues being found]**

> "It found security issues... performance problems... architecture concerns..."

**[Show verification progress]**

> "Now it's generating tests for each issue. If the test FAILS, the bug is VERIFIED."

**[Wait for completion, show results]**

> "8 issues found. 8 verified. 100% verification rate."
>
> "Here's a verified security issue—the test proved it exists. And here's the AI-generated fix."

## Key Features (1:50 - 2:20)

**[Show: Results detail]**

> "Each finding includes:
> - The problematic code
> - A recommendation
> - The verification test that proved it
> - An AI-generated fix ready to copy"

**[Show: Export button / Markdown output]**

> "Export the full report as Markdown for your team."

## Technical Highlights (2:20 - 2:45)

**[Show: /health endpoint or architecture]**

> "Under the hood:
> - Gemini 3 Pro with 2M context—no RAG needed
> - E2B sandboxes for safe test execution
> - Parallel file fetching for speed
> - Real-time streaming with SSE"

## Closing (2:45 - 3:00)

**[Show: Title card with URLs]**

> "This is Vibe Engineering—AI that verifies its own work."
>
> "When we ran this on our own codebase, we found and fixed real bugs."
>
> "Gemini Code Doctor. Try it at the link below."
>
> "Thank you."

---

## Recording Notes

- **Demo repo:** Use the agent's own repo (self-analysis is impressive)
- **Fallback:** Have the markdown export ready as static content
- **Key moment:** The 100% verification rate (8/8) is the "wow" factor
- **Timing:** Self-analysis demo ~60s, features ~30s, keeps under 3 min
- **URLs to show:**
  - Demo: https://gemini-frontend-murex.vercel.app
  - API: https://gemini-agent-hackathon-production.up.railway.app
  - GitHub: https://github.com/languageseed/gemini-agent-hackathon
