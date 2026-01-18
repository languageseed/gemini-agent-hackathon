# Deployment, Security & Compliance Guide

> For Gemini 3 Hackathon Projects

## 🚀 Where to Host Your Project

### Required Deliverables

| Deliverable | Platform | Requirements |
|-------------|----------|--------------|
| **Code Repository** | GitHub | Must be PUBLIC |
| **Live Demo** | See options below | Must work WITHOUT login |
| **Demo Video** | YouTube/Vimeo/Loom | Unlisted is OK |

### Hosting Options Comparison

#### Option 1: Google AI Studio (Recommended for Prototypes)
```
✅ Free tier
✅ Integrated with Gemini API
✅ No hosting setup needed
✅ Judges familiar with it
⚠️ Limited customization
```
**Best for:** Marathon Agent, Real-Time Teacher, Vibe Engineering prototypes

#### Option 2: Vercel (Recommended for Web Apps)
```bash
# Deploy from terminal
npm i -g vercel
vercel --prod

# Or connect GitHub for auto-deploy
```
**Pros:** Free tier, auto SSL, instant deploys, serverless functions
**Best for:** Creative Autopilot with web UI, React/Next.js apps

#### Option 3: Railway (Recommended for Python/FastAPI)
```bash
# Deploy with railway.app
# 1. Connect GitHub repo
# 2. Set environment variables in dashboard
# 3. Auto-deploys on push
```
**Pros:** Docker support, databases, easy Python deploy
**Best for:** Full-stack Marathon Agents, APIs with databases

#### Option 4: Google Cloud Run (Production-Grade)
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/app
gcloud run deploy --image gcr.io/PROJECT_ID/app --platform managed
```
**Pros:** Scales to zero (cheap), production-ready, GCP ecosystem
**Best for:** Complex autonomous agents, GPU requirements

---

## 🔐 Security Checklist

### Before Making Repo Public

```bash
# Check for secrets in git history
git log --all --full-history -- "**/.*" "**/*.env*" "**/secret*" "**/key*"

# Scan for secrets (install gitleaks first)
gitleaks detect --source . -v

# Remove secrets from history if found
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/secret/file' \
  --prune-empty --tag-name-filter cat -- --all
```

### API Key Management

```markdown
❌ NEVER:
- Commit API keys to git (even if you delete later - they're in history!)
- Hardcode keys in frontend JavaScript (visible to users)
- Share keys in demo videos
- Use production keys in public demos

✅ ALWAYS:
- Use environment variables
- Use a secrets manager (Railway/Vercel/GCP Secret Manager)
- Rotate keys after any accidental exposure
- Use restricted API keys with minimal permissions
```

### Environment Variables Pattern

```python
# src/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Secure configuration - never commit actual values!"""
    
    # Gemini API
    gemini_api_key: str = ""  # Set via GEMINI_API_KEY env var
    
    # Application
    debug: bool = False
    allowed_origins: list[str] = ["https://your-app.vercel.app"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### Frontend Security (If Applicable)

```javascript
// ❌ NEVER hardcode API keys in frontend source code
const apiKey = "sk-abc123..."  // NEVER DO THIS!

// ✅ PRODUCTION: Call your own backend, which has the key securely
const response = await fetch('/api/generate', {
    method: 'POST',
    body: JSON.stringify({ prompt: userInput })
});
```

#### Demo/Hackathon Exception

For hackathon demos, it's acceptable to have users enter their own API key (stored in sessionStorage):

```javascript
// ⚠️ DEMO ONLY - user-provided key stored in sessionStorage
// This is for hackathon judging where users test with their own keys
// NOT recommended for production - use server-side auth instead

// sessionStorage is preferred over localStorage:
// - Cleared when tab closes
// - Not persisted across sessions
// - Reduces exposure window
```

**Our demo uses this pattern because:**
1. Judges may want to test with their own API keys
2. We don't want to expose our API quota to public abuse
3. sessionStorage limits exposure (cleared on tab close)

**For production:** Always use server-side authentication and never expose API keys to the frontend.

---

## ✅ Hackathon Compliance Checklist

### Content Restrictions

```markdown
❌ NOT ALLOWED:
- [ ] Medical/mental health diagnostics or advice
- [ ] Content that could harm minors
- [ ] Discriminatory content
- [ ] Privacy-violating data collection
- [ ] Malicious code or security exploits

✅ REQUIRED:
- [ ] Original work (not previously submitted elsewhere)
- [ ] Uses Gemini API (not just Gemini in name)
- [ ] Demonstrates actual functionality (not mockups)
- [ ] Accessible without login/authentication
```

### Technical Compliance

```markdown
- [ ] Public GitHub repository
- [ ] Working demo URL (no 404s, no login walls)
- [ ] Demo video under 3 minutes
- [ ] ~200 word Gemini integration description
- [ ] Code actually runs (judges will test!)
```

### Intellectual Property

```markdown
- [ ] You own or have rights to all code
- [ ] Third-party code is properly attributed
- [ ] No copyrighted content without permission
- [ ] Open source licenses are respected
```

---

## 📁 Recommended Project Structure for Deployment

```
your-hackathon-project/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Auto-deploy on push
├── src/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Pydantic settings
│   └── ...
├── .env.example                # Template (commit this)
├── .gitignore                  # Include .env, __pycache__, etc.
├── Dockerfile                  # For container deployment
├── requirements.txt            # Pinned dependencies
├── README.md                   # Setup instructions
└── vercel.json / railway.toml  # Platform-specific config
```

### .gitignore (Essential)

```gitignore
# Environment files (NEVER commit .env)
.env
.env.local
.env.*.local

# API keys
*.pem
*.key
secrets/

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## 🔧 Platform-Specific Setup

### Vercel (vercel.json)

```json
{
  "version": 2,
  "builds": [
    { "src": "src/main.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "src/main.py" }
  ]
}
```

### Railway

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn src.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
```

### Google Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/app', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/app']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'app'
      - '--image=gcr.io/$PROJECT_ID/app'
      - '--platform=managed'
      - '--region=us-central1'
      - '--allow-unauthenticated'
```

---

## 🎬 Demo Video Best Practices

### Hosting Options
- **YouTube** (unlisted) - Most reliable
- **Vimeo** - Good quality
- **Loom** - Easy recording + hosting
- **Google Drive** - Works but less reliable embedding

### Video Checklist

```markdown
- [ ] Under 3 minutes (judges may not watch longer!)
- [ ] Clear audio (use a microphone)
- [ ] Show the problem (first 30 seconds)
- [ ] Show the solution working (main content)
- [ ] Show architecture diagram briefly
- [ ] No sensitive data visible
- [ ] Test the link works publicly
```

---

## 🚨 Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| API key in repo | Key leaked, account compromised | Use env vars, scan with gitleaks |
| Demo requires login | Judges can't test | Add public/guest mode |
| Demo URL is localhost | Only works on your machine | Deploy to cloud platform |
| Video too long | Judges skip it | Edit to under 3 min |
| Broken demo on deadline | Can't submit | Test 1 day before |
| Private GitHub repo | Judges can't review code | Make public before deadline |

---

## ⏰ Pre-Submission Deployment Checklist

### 1 Week Before Deadline

```markdown
- [ ] Demo deployed and stable
- [ ] GitHub repo organized and documented
- [ ] README has clear setup instructions
- [ ] All secrets removed from git history
- [ ] Video recorded and uploaded
```

### Day Before Deadline

```markdown
- [ ] Test demo URL in incognito browser
- [ ] Test demo on mobile device
- [ ] Verify video link is accessible publicly
- [ ] Verify GitHub repo is PUBLIC
- [ ] Write 200-word Gemini description
```

### Submission Day

```markdown
- [ ] Final smoke test of demo
- [ ] Submit early (don't wait until 4:59 PM!)
- [ ] Save confirmation/screenshot
- [ ] Celebrate 🎉
```
