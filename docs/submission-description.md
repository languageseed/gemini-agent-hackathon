# Gemini Code Doctor - Gemini Integration Description

**Word Count: ~200**

---

## How We Use Gemini

**Gemini Code Doctor** is a "Vibe Engineering" agent that analyzes codebases and **proves its findings through automated testing**—not just reporting issues, but verifying they exist.

### Deep Gemini Integration

**1. Large-Context Analysis (No RAG)**
We clone GitHub repositories and send full source code to Gemini 3 Pro's 2M token context window. No chunking, no retrieval—holistic understanding of code relationships across all files.

**2. Multi-Phase Orchestration**
The agent runs three analysis phases: security scanning (secrets, vulnerabilities), verified code analysis (bugs confirmed via tests), and evolution advice (architecture improvements). This is autonomous multi-step execution—not a single prompt.

**3. Automated Verification (The "Vibe" in Vibe Engineering)**
For each detected issue, Gemini generates a self-contained Python test designed to FAIL if the bug exists. Tests execute in E2B sandboxes. We classify: assertion failures = verified bugs, test passes = possible false positive.

**4. AI-Generated Fixes**
For verified bugs, Gemini proposes targeted fixes with explanations—ready to copy into your codebase.

### Why This Matters

Traditional analyzers produce false positives. We achieved **100% verification rate** (8/8 issues confirmed) when the agent analyzed its own codebase. AI that proves its own work.

---

**URLs:**
- Demo: https://gemini-frontend-murex.vercel.app
- API: https://gemini-agent-hackathon-production.up.railway.app
- GitHub: https://github.com/languageseed/gemini-agent-hackathon
