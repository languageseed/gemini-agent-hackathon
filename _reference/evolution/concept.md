# Evolver – Concept & Discussion

**Status:** Discussion / Pre-build  
**Purpose:** AI service that reviews platform artifacts and recommends next features, enhancements, and improvements per app, infra, or pipeline. Optionally runs your existing elaboration/fine-tuning prompts **passively** (e.g. overnight) like automated testing and review.

---

## 1. What Evolver Is

**Evolver** is an AI-driven “platform evolution advisor” that:

- **Ingests**: Containers (Dockerfiles, compose), code, specifications, plans, backlogs, defects, pipeline configs, and standards.
- **Evaluates**: Consistency, gaps, technical debt, alignment with `APP_STRUCTURE_STANDARD`, `APPLICATION-POLICY`, and `PROJECT_ALIGNMENT_GUIDE`.
- **Outputs**: Prioritized recommendations for the next features and enhancements—per app, per infra component, or per pipeline—with rationale and optional concrete tasks.

It sits **outside** the main platform so it can reason over the whole monorepo and tooling without being tied to any one app’s release cycle or runtime.

---

## 2. Inputs (What Evolver Reads)

| Category | Examples |
|----------|----------|
| **Containers** | `Dockerfile`, `docker-compose.yml`, Traefik routes, resource limits |
| **Code** | App source trees, entry points, dependency files |
| **Specifications** | `spec/SPECIFICATION.md`, `spec/api/`, `spec/deployment/`, `spec/integration/` |
| **Plans & backlogs** | `spec/backlog/`, `ENHANCEMENTS.md`, `BACKLOG.md`, roadmap docs |
| **Defects** | Issue trackers, post-mortems, known limitations in specs |
| **Pipeline & infra** | Woodpecker, deploy scripts, DNS/TLS/routing from policy |
| **Standards** | `APP_STRUCTURE_STANDARD`, `APPLICATION-POLICY`, `PROJECT_ALIGNMENT_GUIDE`, cursor-standards |

Evolver doesn’t need to *change* these; it needs read-only access (and possibly a small amount of structured output written to a dedicated location, e.g. `evolver/reports/`).

---

## 3. Outputs (What Evolver Produces)

- **Per-app**: “Next features/enhancements” with priority, rationale, and optional link to spec/backlog.
- **Per-infra**: Gaps in DNS/TLS/routing/secrets/logging vs policy; suggested next steps.
- **Per-pipeline**: CI/deploy improvements, security/scans, performance.
- **Cross-cutting**: Alignment with standards, missing specs, inconsistent patterns.

Formats could be: Markdown reports, structured JSON/YAML for tooling, or both.

---

## 4. Why “Outside” the Platform?

- **Neutral view**: No dependency on any single app’s stack or release train.
- **Single codebase view**: Can assume “whole repo” context (e.g. clone or mount of `platform`).
- **Simpler ops**: Can run as a scheduled job or on-demand script without being part of Traefik/Vault/seed deployment.
- **Cursor CLI fit**: Cursor Agent/CLI works best with a filesystem and terminal; a separate “evolver” repo or a `apps/evolver/docs/` runner keeps that boundary clear.

You can still version Evolver itself (e.g. in-repo under `apps/evolver/docs/` or in a separate repo) and run it from a developer machine or a single CI job that has repo access.

---

## 4a. Relationship to Monocle (pipeline-monocle-security-agent)

**Monocle** (`infra/pipeline-monocle-security-agent`) is the platform’s security auditing and compliance pipeline: agentless scanning (Docker socket, SSH, API probing), secret/CVE/container/network/API checks, compliance frameworks (NIST, CIS, PCI-DSS, etc.), zone-based assessment, AI-enhanced findings, and **CIP (Continuous Improvement Plan)** backlog. It runs as a service (API + UI) and produces audits, reports, and CIP items.

**Evolver** is a separate, complementary layer:

| Dimension | Monocle | Evolver |
|-----------|---------|---------|
| **Focus** | Security & compliance (CVEs, secrets, hardening, frameworks) | Evolution & product health (alignment, spec/code/container drift, next features, backlog, architecture docs) |
| **Inputs** | Running containers, hosts via SSH, registered projects, scan policies | Repo (code, specs, Dockerfiles, standards); optionally live state via SSH |
| **Outputs** | Audits, compliance reports (HTML), findings, CIP backlog, dashboards | Structured Markdown report: alignment gaps, dependencies, pending work, spec/architecture proposals |
| **Execution** | Service (API 8690, UI); scheduled or on-demand audits | Scheduled or on-demand job (Cursor CLI + prompts); no long-lived service required |

**Keep them separate.**

There is a **conflict of interest** between the two: Monocle’s role is security and compliance (clear, auditable findings; CIP as security/compliance backlog); Evolver’s role is evolution and prioritization (features, alignment, spec updates). Combining them—e.g. feeding Monocle’s CIP or findings into Evolver’s “next actions”—would mix security remediation with product backlog and could bias prioritization, blur ownership of recommendations, or dilute the independence of security findings. Security and compliance should stay in Monocle’s lane; evolution and product health in Evolver’s.

- **No integration:** Do not feed Monocle data into Evolver (or the reverse). Run each on its own schedule; consume their outputs separately (Monocle: audits and CIP; Evolver: evolution reports).
- **Different questions:** Monocle: “Is this secure and compliant?” Evolver: “Is this aligned, and what should we build or fix next?” Same repo and containers can be *inputs* to both, but their outputs and ownership remain distinct.

---

## 5. Using Cursor CLI as the Engine

**Idea:** Use Cursor’s Agent via CLI to drive the “review and recommend” workflow.

**Pros:**

- **Already aligned with how you work**: Same models and context (rules, MCP, semantic search) as in the editor.
- **Non-interactive mode**: `agent -p "..." --output-format text` fits scripts and CI; you can pass a carefully crafted “Evolver prompt” and parse or store the output.
- **Plan mode**: `/plan` or `--mode=plan` can be used to get a structured “what to evaluate” before running a full Agent pass.
- **Ask mode**: Read-only exploration (`--mode=ask`) for safe, no-edit audits.
- **Context**: Can use `@codebase`, rules in `.cursor/rules` (or evolver-specific rules), and MCP for external data (e.g. Linear, GitHub issues) if you add them later.
- **Sessions**: `agent resume` could support multi-step “conversations” for deeper dives on one app.

**Cons / considerations:**

- **CLI is beta**: Security and behavior may change; use in trusted environments only.
- **Approval flows**: In interactive mode the agent may prompt for approval; for full automation you’ll want non-interactive (print) mode and clear prompts so it doesn’t need to run destructive commands.
- **Determinism**: Output may vary run-to-run; you may want to version prompts and optionally snapshot outputs for comparison.
- **Scope control**: You need to constrain what the agent reads (e.g. only `apps/`, `standards/`) so it doesn’t wander; a dedicated evolver rule/skill can state “only read these paths and only output a report”.
- **Cost/latency**: Each run consumes model calls; scheduling (e.g. weekly) and scoped prompts help.

**Verdict:** Cursor CLI is a strong candidate for the “brain” of Evolver: one or more prompts (and optionally rules/MCP) define the evaluation; the CLI runs in non-interactive mode and produces reports. You can wrap it in a small runner script (or later a tiny service) that sets working directory, env, and output paths.

---

## 6. Passive Overnight: Same Prompts, Automated Review

**Goal:** Reuse the same prompts you use for elaborating or fine-tuning code, but run them passively (e.g. overnight) so you get automated testing-and-review style results without being at the keyboard.

### Why a "service" helps

- **Consistency:** One place to store and version prompts; no copy-paste.
- **Scheduling:** Run at a fixed time (e.g. 2am) so the job finishes before you start work.
- **Artifacts:** Reports and diffs in a known location (or CI artifacts) instead of buried in chat.
- **Repeatability:** Same prompts, same scope, comparable runs over time.

### Methods compared

| Method | Same prompts & context | Truly passive / scheduled | Overnight without your machine | Best for |
|--------|------------------------|---------------------------|---------------------------------|----------|
| **Cursor CLI + cron (local)** | Yes: rules, codebase, MCP | Yes: cron at 2am | No: machine must be on | Your own laptop/desktop; full control. |
| **Cursor CLI on seed (Linux)** | Yes if repo + rules on seed | Yes: cron or systemd timer | Yes: seed is already on 24/7 | Single host; no extra CI runner; install CLI + set `CURSOR_API_KEY`. |
| **Cursor CLI + CI (Woodpecker/GHA)** | Yes if runner has CLI + repo + auth | Yes: pipeline schedule | Yes: runner runs at night | True "nightly job"; reports as pipeline artifacts. |
| **Cursor Cloud Agent** | Yes: same Cursor context | Trigger-by-hand (e.g. "& run this") | Yes: runs in Cursor's cloud | "Start before bed, check at cursor.com/agents in morning"; not cron. |
| **CodeRabbit / PR review SaaS** | No: their prompts, not yours | Yes: scheduled reports (e.g. weekly) | Yes | Standardized PR review; not your elaboration prompts. |
| **Custom service (OpenAI/Claude API)** | You rebuild context (code, rules) | Yes: your scheduler | Yes | Full control; more work to mirror Cursor's context and tool use. |
| **Devin / other agent APIs** | No: different product | Often PR/event-triggered | Yes | Alternative stack; not your existing Cursor prompts. |

### Recommendation for "same prompts, passive overnight"

1. **Best fit: Cursor CLI in a scheduled job**
   - **Local (macOS):** Use cron or launchd to run your runner script (e.g. `evolve.sh`) with print mode. Prompts in `apps/evolver/docs/prompts/`; output in `evolver/out/` or a timestamped file.
   - **On seed (Linux):** Cursor CLI is available on Linux (`curl https://cursor.com/install -fsS | bash`; binary in `~/.local/bin`). On your seed host, install the CLI, set `CURSOR_API_KEY` (from [Cursor dashboard](https://cursor.com/dashboard)), ensure the platform repo is present (clone or mount), and run the same evolve script via **cron** or a systemd timer. Reports can live under the repo (e.g. `evolver/out/`) or a fixed path; no need for a separate CI runner if seed already has the repo and is on 24/7.
   - **Via CI (Woodpecker/GHA):** Runner installs Cursor CLI, has `CURSOR_API_KEY`, clones repo, runs the script on a schedule. Artifacts = nightly reports. Use this when you prefer not to run the CLI on seed or want reports as pipeline artifacts.
   - **Caveat:** Some users report the CLI occasionally hanging in non-interactive mode. Wrap the call in a **timeout** (e.g. `timeout 3600 agent -p "..."`) so cron/CI doesn't block indefinitely.

2. **Alternative: Cloud Agent as "run before bed"**
   - If you don't need strict "every night at 2am", you can trigger a Cloud Agent (e.g. "& run evolution report for apps/journey using prompts in apps/evolver/docs") before leaving. It runs in Cursor's cloud; you check results at cursor.com/agents in the morning. Same prompts and context, but not scheduler-driven.

3. **When to consider a custom LLM service**
   - If you need runs on a server that can't run Cursor CLI (e.g. no GUI, strict air-gap), or you want to own the entire pipeline and are willing to replicate context (file trees, rules, tool use) via API, then a small service calling OpenAI/Anthropic (or similar) is an option. More build and maintenance; only worth it if CLI/Cloud Agent don't meet your constraints.

**Summary:** For reusing your elaboration/fine-tuning prompts passively overnight, **Cursor CLI in a scheduled job** (cron locally or CI nightly) is the best fit. Use Cloud Agent for "start before bed, check in morning" without running your own scheduler. Other methods either don't use your prompts or require rebuilding context elsewhere.

---

## 7. Minimal Design Before Building

1. **Placement**
   - **Option A:** Repo under `apps/evolver/docs/` (prompts, runner script, schema for outputs). Evolver “code” is just prompts + shell (and maybe a small parser).
   - **Option B:** Separate repo that clones or mounts `platform` and runs the same scripts. Keeps platform repo free of automation tooling.

2. **Runner**
   - Single entry point, e.g. `./evolve.sh [app-name|infra|pipeline|all]`.
   - Script sets `PLATFORM_ROOT`, invokes `agent -p "$(cat prompts/evaluate-app.txt)" ...` (or similar) with `--output-format text`, writes to `evolver/out/` or a timestamped file.
   - Optional: second pass with `--mode=plan` to get a plan, then Agent to fill in the report.

3. **Prompts**
   - One prompt (or prompt template) per scope. Example: `prompts/review-product.txt` (see below).
   - Each prompt should state: (a) scope and target (e.g. seed/sapling, SSH); (b) items to review; (c) standards to check against; (d) output format (Markdown sections, or JSON if you add a parser).
   - **Example prompt** (full template in `prompts/review-product.txt`): review a product (containers, code, specs, plans, deployment pipeline); target seed and sapling via SSH; assess alignment to `standards/`, build environment context, review container/config/code/spec, identify dependencies and enhancements/defects/next actions, assess persistence across restart/rebuild; do not restart or rebuild containers unless zero data-loss impact; update specs and architecture markdowns with latest info and Mermaid diagrams. Substitute `{{PRODUCT}}` with the app name (e.g. journey). Use `--force` only if the agent should apply spec/architecture edits; otherwise it proposes in the report.

4. **Output**
   - Start with Markdown reports. Add a simple schema (e.g. “Next features”, “Gaps”, “Risks”) so you can later parse or feed into Linear/backlog if desired.

5. **No new service yet**
   - No API, no container. Just CLI + scripts + prompts. If you later want a “service”, it could be a thin wrapper that runs the same script on a schedule and stores reports (e.g. in S3 or in-repo).

---

## 8. Next Steps

- **Decide placement**: In-repo `apps/evolver/docs/` vs separate repo.
- **Prompts**: e.g. “Evaluate app X: read `apps/X/` and `standards/`, list next 5 features/enhancements with rationale; output Markdown only.”
- **Run once by hand**: From repo root, `agent -p "$(sed 's/{{PRODUCT}}/journey/g' apps/evolver/docs/prompts/review-product.txt)"` (add `--output-format text` and redirect to `apps/evolver/docs/out/` if desired). Use `--force` only if you want the agent to apply spec/architecture updates.
- **Add a runner script**: e.g. `evolve.sh [app-name]` that substitutes `{{PRODUCT}}`, invokes the CLI with a timeout, and writes to a fixed output dir.
- **Iterate**: Add rules or an evolver-specific skill so the agent stays on task and format; then consider scheduling or a small “report collector” if useful.

If you’d like, next we can add a first prompt and a minimal `evolve.sh` in `apps/evolver/docs/` and try it on one app (e.g. journey or service-3-content-processor) without committing to a full build.
