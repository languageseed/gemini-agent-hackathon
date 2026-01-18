# Verified Codebase Analyst - Gemini Integration Description

**Word Count: ~200**

---

## How We Use Gemini

**Verified Codebase Analyst** leverages Gemini 3's 2-million token context window to analyze codebases without chunking or RAG—enabling holistic code understanding.

### Deep Integration Points

**1. Large-Context Analysis**
We clone GitHub repositories and send source code to Gemini 3 Pro Preview. This enables understanding of code relationships across multiple files in a single prompt—no RAG needed.

**2. Structured Issue Extraction**
Gemini identifies bugs, security vulnerabilities, and performance issues, outputting structured JSON with severity levels, affected files, and line numbers.

**3. Automated Verification (Vibe Engineering)**
For each detected issue, Gemini generates a self-contained Python test designed to FAIL if the bug exists. Tests execute in E2B sandboxes. We distinguish assertion failures (verified bugs) from runtime errors (inconclusive).

**4. AI-Generated Fix Proposals**
For verified bugs, Gemini generates targeted fix proposals with explanations.

**5. Programmatic Access**
A Python client (`gemini-analyst`) enables CI/CD integration with async job submission, webhooks, and status polling—bringing verified analysis into automated pipelines.

### Why This Matters

Traditional static analysis produces false positives. By having Gemini write and execute verification tests, we add confidence to findings through "Vibe Engineering"—AI that proves its own work.

---

**URLs:**
- Demo: https://gemini-frontend-murex.vercel.app
- GitHub: https://github.com/languageseed/gemini-agent-hackathon
