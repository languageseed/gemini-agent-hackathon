#!/bin/bash
# Run the backend locally for testing

set -e

cd "$(dirname "$0")/.."

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from env.example..."
    cp env.example .env
    echo "📝 Edit .env with your API keys before continuing"
    exit 1
fi

# Source env vars
set -a
source .env
set +a

# Check required vars
if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" == "your_gemini_api_key_here" ]; then
    echo "❌ GOOGLE_API_KEY not set in .env"
    exit 1
fi

echo "🚀 Starting local server..."
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Run with auto-reload for development
uvicorn src.main:app --reload --host 0.0.0.0 --port ${PORT:-8000}
