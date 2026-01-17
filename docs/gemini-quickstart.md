# Gemini API Quickstart

> Connect your Google AI Studio account to your Cursor project

## 🚀 Quick Setup (5 minutes)

### 1. Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com)
2. Click **"Get API Key"** in the left sidebar
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

### 2. Set Up Environment

```bash
# In your project directory
cp env.example .env
```

Then edit `.env`:
```bash
GOOGLE_API_KEY=AIzaSy...your-actual-key...
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Install the SDK

**Python:**
```bash
pip install google-generativeai
```

**Node.js/TypeScript:**
```bash
npm install @google/generative-ai
```

---

## 🐍 Python Usage

### Basic Text Generation

```python
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Create a model instance
model = genai.GenerativeModel("gemini-2.0-flash")

# Generate content
response = model.generate_content("Explain quantum computing in simple terms")
print(response.text)
```

### Streaming Responses (Better UX)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# Stream the response
response = model.generate_content(
    "Write a short story about a robot",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

### Multi-turn Conversation (Chat)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# Start a chat session
chat = model.start_chat(history=[])

# First message
response = chat.send_message("Hi! I'm building a hackathon project.")
print(response.text)

# Follow-up (maintains context)
response = chat.send_message("What are some innovative AI project ideas?")
print(response.text)

# Access full history
print(chat.history)
```

### Vision (Image Analysis)

```python
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# From file
image_path = Path("screenshot.png")
image_data = {
    "mime_type": "image/png",
    "data": image_path.read_bytes()
}

response = model.generate_content([
    "Describe what you see in this image:",
    image_data
])
print(response.text)
```

### Function Calling (Tool Use)

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define tools the model can use
tools = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }
]

model = genai.GenerativeModel("gemini-2.0-flash", tools=tools)

response = model.generate_content("What's the weather in San Francisco?")

# Check if model wants to call a function
for candidate in response.candidates:
    for part in candidate.content.parts:
        if hasattr(part, 'function_call'):
            print(f"Function: {part.function_call.name}")
            print(f"Args: {part.function_call.args}")
```

---

## 📦 TypeScript/Node.js Usage

### Basic Text Generation

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function generate() {
    const result = await model.generateContent("Explain quantum computing");
    console.log(result.response.text());
}

generate();
```

### Streaming

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function streamGenerate() {
    const result = await model.generateContentStream("Write a haiku about coding");
    
    for await (const chunk of result.stream) {
        process.stdout.write(chunk.text());
    }
}

streamGenerate();
```

### Chat

```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function chat() {
    const chatSession = model.startChat({
        history: [],
    });

    const result1 = await chatSession.sendMessage("Hello!");
    console.log(result1.response.text());

    const result2 = await chatSession.sendMessage("What can you help me build?");
    console.log(result2.response.text());
}

chat();
```

---

## ⚡ FastAPI Integration

### Complete Example

```python
# src/main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

app = FastAPI(title="Gemini Hackathon API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000

class GenerateResponse(BaseModel):
    text: str
    usage: dict | None = None

@app.get("/health")
async def health():
    return {"status": "healthy", "model": os.environ.get("GEMINI_MODEL")}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        response = model.generate_content(
            request.prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=request.max_tokens
            )
        )
        return GenerateResponse(
            text=response.text,
            usage={"candidates": len(response.candidates)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
```

### Run It

```bash
# Install dependencies
pip install fastapi uvicorn google-generativeai python-dotenv

# Run the server
uvicorn src.main:app --reload
```

### Test It

```bash
# Health check
curl http://localhost:8000/health

# Generate content
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about AI"}'
```

---

## 🎛️ Available Models

| Model | Best For | Context Window |
|-------|----------|----------------|
| `gemini-2.0-flash` | Fast responses, general use | ~1M tokens |
| `gemini-2.0-flash-thinking` | Complex reasoning with thinking | ~1M tokens |
| `gemini-1.5-pro` | Complex tasks, long context | ~2M tokens |
| `gemini-1.5-flash` | Balanced speed/quality | ~1M tokens |

> **Note:** Gemini 3 models may be named differently when released. Check [AI Studio](https://aistudio.google.com) for latest model names.

---

## 🔧 Configuration Options

### Generation Config

```python
from google.generativeai import GenerationConfig

config = GenerationConfig(
    temperature=0.7,          # Creativity (0.0 - 2.0)
    top_p=0.95,              # Nucleus sampling
    top_k=40,                # Top-k sampling
    max_output_tokens=2000,  # Response length limit
    stop_sequences=["END"],  # Stop generation at these
)

response = model.generate_content(
    "Write a story",
    generation_config=config
)
```

### Safety Settings

```python
from google.generativeai import types

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "BLOCK_ONLY_HIGH"
    }
]

response = model.generate_content(
    "Your prompt",
    safety_settings=safety_settings
)
```

---

## 🐛 Common Issues

### "API key not valid"
```bash
# Check your key is set
echo $GOOGLE_API_KEY

# Make sure .env is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('GOOGLE_API_KEY', 'NOT SET')[:10])"
```

### "Model not found"
```python
# List available models
for model in genai.list_models():
    print(model.name)
```

### Rate Limits (429 errors)
```python
import time
from google.api_core import retry

# Add retry logic
@retry.Retry(predicate=retry.if_exception_type(Exception))
def generate_with_retry(prompt):
    return model.generate_content(prompt)
```

---

## 📚 Resources

- [Gemini API Docs](https://ai.google.dev/docs)
- [Python SDK Reference](https://ai.google.dev/api/python/google/generativeai)
- [AI Studio](https://aistudio.google.com) - Test prompts in browser
- [Cookbook Examples](https://github.com/google-gemini/cookbook)
