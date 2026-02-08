# Hackathon Compliance Guide

When porting code from the security and evolution reference components, follow these substitutions to ensure compliance with the Gemini 3 Hackathon requirements.

---

## Key Rules

1. **Use Gemini 3 API ONLY** - No OpenAI, Anthropic, or other AI models
2. **Vibe Engineering Track** - Agents that write AND verify code (already aligned)
3. **Not a Wrapper** - Must have autonomous capabilities, multi-step orchestration
4. **Use Gemini's Unique Features** - 1M+ context window, native function calling

---

## Substitution Table

### From Security Components

| Original Component | Issue | Replace With | Status |
|-------------------|-------|--------------|--------|
| `Valet embeddings API` in `compliance_mapper.py` | Uses internal Valet service | Gemini embeddings API or remove | ❌ Needs update |
| `Valet reranking` in `compliance_mapper.py` | Uses internal Valet service | Gemini for semantic matching | ❌ Needs update |
| `AI confidence scoring` | Uses Valet | Gemini verification | ❌ Needs update |
| **Secret patterns** (`secret_scanner.py`) | Pure regex + entropy | Keep as-is | ✅ Compliant |
| **Container checks** (`container_scanner.py`) | Docker inspection | Keep as-is | ✅ Compliant |
| **Finding model** (`models/finding.py`) | Data structure only | Keep as-is | ✅ Compliant |
| **Recommendations** (`recommendations.py`) | Rule-based advice | Keep as-is | ✅ Compliant |
| **Trivy/Grype scanning** | External security tools | Keep as-is (allowed) | ✅ Compliant |

### From Evolution Components

| Original Component | Issue | Replace With | Status |
|-------------------|-------|--------------|--------|
| `gpt-5.2-codex` in `scheduler.py` | Wrong model reference | `gemini-3-pro` | ❌ Needs update |
| Cursor CLI engine | Uses various models | Gemini 3 Pro direct API | ❌ Needs update |
| **Review prompts** (`review_prompt.txt`) | Prompt template only | Adapt for Gemini system prompt | ✅ Compliant |
| **Evolution concept** | Architecture reference | Use as design guide | ✅ Compliant |

---

## Gemini API Patterns to Use

### Embeddings (Replace Valet)

```python
# DON'T use Valet embeddings
# DO use Gemini's native understanding

# Option 1: Use Gemini's semantic understanding directly
async def semantic_match_with_gemini(finding: str, controls: list[str]) -> list[tuple[str, float]]:
    """Use Gemini to semantically match findings to controls."""
    prompt = f"""Given this security finding:
{finding}

Rank these compliance controls by relevance (0.0-1.0):
{json.dumps(controls, indent=2)}

Return JSON: [{{"control": "...", "relevance": 0.X, "reason": "..."}}]"""

    model = genai.GenerativeModel("gemini-3-flash")  # Fast for matching
    response = await model.generate_content_async(prompt)
    return parse_relevance_scores(response.text)
```

### AI Analysis (Replace Valet)

```python
# DON'T use Valet for analysis
# DO use Gemini 3 Pro

async def analyze_with_gemini(codebase: str, focus: str) -> list[Finding]:
    """Use Gemini 3 Pro for codebase analysis."""
    model = genai.GenerativeModel(
        "gemini-3-pro",  # Use Pro for complex reasoning
        system_instruction=ANALYZER_SYSTEM_PROMPT
    )
    
    response = await model.generate_content_async(
        f"Analyze this codebase for {focus} issues:\n\n{codebase}",
        generation_config={"temperature": 0.2}  # Low for accuracy
    )
    return parse_findings(response.text)
```

### Verification (Existing - Keep)

```python
# Already using E2B for code execution - KEEP THIS
# E2B is allowed as external service for sandboxed code execution
```

---

## Model Selection

| Use Case | Model | Why |
|----------|-------|-----|
| Complex analysis | `gemini-3-pro` | Deep reasoning, full codebase |
| Quick matching | `gemini-3-flash` | Fast semantic matching |
| Verification | E2B + Gemini | Code execution in sandbox |
| Evolution advice | `gemini-3-pro` | Strategic recommendations |

---

## What Makes This Compliant

### Vibe Engineering Track ✅

> "Agents that don't just write code but verify it"

- **Analyze**: Gemini identifies issues (security, quality, evolution)
- **Verify**: E2B executes generated tests to confirm bugs
- **Fix**: Gemini proposes fixes for verified issues
- **Orchestrate**: Multi-step pipeline, not single prompt

### Uses Gemini's Unique Features ✅

- **1M+ context**: Full codebase in single context (no chunking/RAG needed)
- **Native function calling**: Tool use for scanning/verification
- **Reasoning**: Complex security and evolution analysis
- **Streaming**: Real-time progress updates

### Not a Wrapper ✅

- Multi-step orchestration (scan → analyze → verify → fix → advise)
- Autonomous verification loop
- Multiple analysis modes (security, quality, evolution)
- Marathon Agent capability (continuous monitoring)

---

## Files to Modify When Porting

### compliance_mapper.py (Security)

```python
# BEFORE (Valet)
async def _get_embedding(self, text: str) -> Optional[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{self.valet_url}/v1/embeddings",
            json={"input": text, "model": self.settings.valet_embedding_model}
        )

# AFTER (Gemini semantic matching - no embeddings needed)
async def semantic_match(self, finding: Finding, controls: list[Control]) -> list[tuple[Control, float]]:
    """Use Gemini for semantic matching instead of embeddings."""
    prompt = self._build_matching_prompt(finding, controls)
    response = await self.gemini_model.generate_content_async(prompt)
    return self._parse_matches(response.text, controls)
```

### scheduler.py (Evolution)

```python
# BEFORE (GPT-5.2-codex)
model = schedule.get("model", "gpt-5.2-codex")

# AFTER (Gemini 3)
model = schedule.get("model", "gemini-3-pro")
```

---

## Pre-Integration Checklist

Before porting each component:

- [ ] Does it use any AI model? → Replace with Gemini 3
- [ ] Does it call external AI APIs? → Replace with Gemini API
- [ ] Is it pure Python logic? → Keep as-is
- [ ] Is it a data model? → Keep as-is
- [ ] Does it use allowed external tools (Trivy, E2B, GitHub)? → Keep as-is

---

## Summary

| Category | File | Action |
|----------|------|--------|
| Secret detection | `secret_scanner.py` | Keep (regex/entropy) |
| Container checks | `container_scanner.py` | Keep (Docker inspection) |
| CVE scanning | External (Trivy) | Keep (external tool) |
| Compliance mapping | `compliance_mapper.py` | Replace Valet with Gemini |
| Recommendations | `recommendations.py` | Keep (rule-based) |
| Evolution prompts | `review_prompt.txt` | Adapt for Gemini |
| Scheduler | `scheduler.py` | Change model to `gemini-3-pro` |
