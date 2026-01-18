# Demo Video Script (~3 minutes)

## Opening (0:00 - 0:20)

**[Show: Title card with project name]**

> "What if AI could not only find bugs in your code—but prove they exist?"
>
> "I'm going to show you Verified Codebase Analyst, powered by Gemini 3's 2-million token context window."

## The Problem (0:20 - 0:40)

**[Show: Screenshot of typical static analysis with many warnings]**

> "Traditional code analyzers produce endless false positives. Developers waste hours investigating issues that aren't real."
>
> "What if the AI could verify its findings before reporting them?"

## The Solution (0:40 - 1:00)

**[Show: Architecture diagram]**

> "Verified Codebase Analyst takes a different approach:
> 1. Load the ENTIRE codebase into Gemini 3's context
> 2. Identify potential issues
> 3. Generate tests that would FAIL if the bug exists
> 4. Execute those tests
> 5. Only report VERIFIED bugs with AI-generated fixes"

## Live Demo (1:00 - 2:20)

**[Show: Frontend UI]**

> "Let me show you this in action."

**[Action: Paste a GitHub repo URL, select 'bugs' focus, enable 'Verify Findings']**

> "I'll analyze this open-source project for bugs."

**[Action: Click 'Analyze & Verify']**

> "Watch the live progress..."

**[Show: Observability panel updating in real-time]**

> "You can see:
> - Issues being discovered with severity levels
> - Tests being generated for each issue  
> - Verification status updating—VERIFIED means the test failed, proving the bug exists"

**[Wait for completion, show results]**

> "Here are the results. Let's look at a verified issue..."

**[Action: Expand a verified issue card]**

> "This critical bug was verified. Here's:
> - The problematic code
> - The test that confirmed it
> - And most importantly—the AI-generated fix we can apply"

**[Action: Click 'Copy Fix' button]**

> "One click and the fix is ready to paste."

## Technical Highlights (2:20 - 2:45)

**[Show: Diagnostics endpoint]**

> "Under the hood:
> - Gemini 3 Pro Preview handles the reasoning
> - E2B sandboxes execute tests safely
> - Full observability for debugging
> - SSE streaming for real-time updates"

**[Show: /diagnostics response]**

> "Every component is monitored with latency tracking."

## Closing (2:45 - 3:00)

**[Show: Title card with URLs]**

> "This is Vibe Engineering—AI that verifies its own work."
>
> "Verified Codebase Analyst. Try it at the link below."
>
> "Thank you."

---

## Recording Notes

- Use a repo known to have bugs for impressive demo
- Prepare fallback screenshots in case of API issues
- Keep terminal visible to show SSE events if needed
- Have architecture diagram ready as backup visual
