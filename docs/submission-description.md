# Verified Codebase Analyst - Gemini Integration Description

**Word Count: ~200**

---

## How We Use Gemini

**Verified Codebase Analyst** leverages Gemini 3's 2-million token context window to analyze codebases without chunking or RAG—enabling holistic code understanding.

### Deep Integration Points

**1. Large-Context Analysis**
We clone GitHub repositories and send source code to Gemini 3 Pro Preview (truncated to ~125K tokens for reliability). This enables understanding of code relationships across multiple files in a single prompt.

**2. Structured Issue Extraction**
Gemini identifies bugs, security vulnerabilities, and performance issues, outputting structured JSON with severity levels, affected files, and line numbers.

**3. Automated Verification (Vibe Engineering)**
For each detected issue, Gemini generates a self-contained Python test designed to FAIL if the bug exists. Tests run in E2B sandboxes with the code snippet embedded. We distinguish assertion failures (verified) from runtime errors (inconclusive).

**4. AI-Generated Fix Proposals**
For verified bugs, Gemini generates targeted fix proposals with explanations. Fixes are labeled "proposed" as they haven't been re-tested.

### Why This Matters

Traditional static analysis produces many false positives. By having Gemini write verification tests, we add a layer of confidence to findings through the "Vibe Engineering" approach—while being transparent about verification boundaries.

---

**URLs:**
- Demo: https://gemini-frontend-murex.vercel.app
- GitHub: https://github.com/languageseed/gemini-agent-hackathon
