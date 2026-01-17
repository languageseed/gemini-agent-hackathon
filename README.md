# Gemini Agent

A Gemini-powered agent with tool calling capabilities for the [Gemini 3 Hackathon](https://gemini3.devpost.com/).

## Features

- **Gemini 3 Integration** - Uses latest Gemini API with function calling
- **Agentic Loop** - Multi-turn execution with tool calling
- **Tool Framework** - Extensible tool definitions
- **Streaming Support** - SSE for real-time responses
- **Production Ready** - FastAPI with Railway deployment

## Quick Start

### 1. Set up environment

```bash
# Copy environment template
cp env.example .env

# Edit .env and add your API key
# Get key from: https://aistudio.google.com/apikey
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
uvicorn src.main:app --reload
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Simple generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what can you do?"}'

# Agent with tools
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What time is it? Then calculate 42 * 17."}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/generate` | POST | Simple text generation |
| `/chat` | POST | Multi-turn chat |
| `/agent` | POST | **Agent with tool calling** |
| `/tools` | GET | List available tools |

## Adding Tools

Edit `src/main.py` to add your own tools:

```python
# 1. Define the tool schema
TOOLS = [
    {
        "name": "your_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }
    },
    # ... existing tools
]

# 2. Implement the tool execution
async def execute_tool(name: str, arguments: dict) -> str:
    if name == "your_tool":
        # Your implementation
        return "result"
    # ... existing tools
```

## Deployment

### Railway

```bash
# Login to Railway
railway login

# Initialize project
railway init

# Add PostgreSQL (optional)
railway add --plugin postgresql

# Add Redis (optional)
railway add --plugin redis

# Deploy
railway up

# Set environment variables
railway variables set GOOGLE_API_KEY=your_key
```

### Vercel (Frontend Only)

The API should run on Railway. Deploy a frontend to Vercel that calls the Railway API.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  /agent endpoint                                        │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Agentic Loop                          │   │
│  │                                                  │   │
│  │  1. Send prompt + tools to Gemini               │   │
│  │  2. If tool calls → execute tools               │   │
│  │  3. Send results back to Gemini                 │   │
│  │  4. Repeat until done or max iterations         │   │
│  │                                                  │   │
│  └─────────────────────────────────────────────────┘   │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │    Tool 1   │  │    Tool 2   │  │    Tool N   │    │
│  │ get_time    │  │  calculate  │  │ search_web  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Gemini API   │
                  │  (Cloud)      │
                  └───────────────┘
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key from AI Studio |
| `GEMINI_MODEL` | No | Model name (default: gemini-3-flash-preview) |
| `PORT` | No | Server port (default: 8000) |
| `DEBUG` | No | Enable debug mode (default: false) |
| `DATABASE_URL` | No | PostgreSQL connection (Railway provides) |
| `REDIS_URL` | No | Redis connection (Railway provides) |

## Hackathon Submission

### Judging Criteria

| Criteria | Weight | How This Project Scores |
|----------|--------|------------------------|
| Technical Execution | 40% | Deep Gemini integration with tool calling |
| Innovation | 30% | Autonomous agent, not a wrapper |
| Impact | 20% | Extensible platform for real tasks |
| Presentation | 10% | Clear architecture, demo-ready |

### Submission Checklist

- [ ] ~200 word Gemini integration description
- [ ] Public demo URL (Railway)
- [ ] Public GitHub repository
- [ ] ~3 minute demo video

## License

MIT - Hackathon Project
# Trigger redeploy Sun Jan 18 06:48:49 AWST 2026
