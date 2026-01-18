# Verified Codebase Analyst - Gemini Integration Description

**Word Count: ~200**

---

## How We Use Gemini

**Verified Codebase Analyst** leverages Gemini 3's groundbreaking 2-million token context window to analyze entire codebases in a single prompt—no RAG, no chunking, no context loss.

### Deep Integration Points

**1. Whole-Codebase Analysis**
We clone GitHub repositories and feed the complete source code to Gemini 3 Pro Preview. This enables holistic understanding of code relationships, architectural patterns, and subtle bugs that span multiple files.

**2. Structured Issue Extraction**
Gemini identifies bugs, security vulnerabilities, and performance issues, outputting structured JSON with severity levels, affected files, and line numbers.

**3. Automated Verification (Vibe Engineering)**
For each detected issue, Gemini generates a Python test designed to FAIL if the bug exists. We execute these tests in E2B sandboxes. If the test fails, the bug is VERIFIED—not a false positive.

**4. AI-Generated Fixes**
For verified bugs, Gemini generates targeted code fixes with explanations. The test that verified the bug can validate the fix.

### Why This Matters

Traditional static analysis produces many false positives. By having Gemini write verification tests, we prove bugs exist before reporting them—achieving unprecedented accuracy through the "Vibe Engineering" approach: agents that verify their own work.

---

**URLs:**
- Demo: https://gemini-frontend-murex.vercel.app
- GitHub: https://github.com/languageseed/gemini-agent-hackathon
