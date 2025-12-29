# FRIDA Backend API

FastAPI backend for the FRIDA Command Interpreter web application.

## Features

- RESTful API for command interpretation
- BAML integration for LLM-powered parsing
- Multiple model support (Gemini, GPT-4, Claude, Llama)
- Command generation with CommandGenerator
- API key authentication
- CORS protection
- Health check endpoint

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your keys

# Run server
uvicorn main:app --reload
```

Server runs at `http://localhost:8080`

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy"}
```

### Get Available Models
```
GET /models
Headers: Authorization: Bearer <API_KEY>
Response: {"models": ["GEMINI_FLASH_2_5", ...]}
```

### Interpret Command
```
POST /interpret
Headers:
  Authorization: Bearer <API_KEY>
  Content-Type: application/json
Body: {
  "command": "go to the kitchen and pick up the apple",
  "model": "GEMINI_FLASH_2_5",
  "execute": false
}
Response: {
  "commands": [...],
  "string_command": "...",
  "execution_results": null
}
```

### Generate Random Command
```
POST /generate
Headers:
  Authorization: Bearer <API_KEY>
  Content-Type: application/json
Body: {
  "model": "GEMINI_FLASH_2_5",
  "execute": false
}
Response: {
  "commands": [...],
  "string_command": "...",
  "execution_results": null
}
```

## Environment Variables

```env
BACKEND_API_KEY=<secure-random-key>
OPENROUTER_API_KEY=<your-openrouter-key>
ALLOWED_ORIGINS=http://localhost:3000,https://your-app.vercel.app
PORT=8080
```

## Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --log-level debug
```

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Get models (replace YOUR_KEY)
curl -H "Authorization: Bearer YOUR_KEY" \
     http://localhost:8080/models

# Interpret command
curl -X POST http://localhost:8080/interpret \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "go to the kitchen",
    "model": "GEMINI_FLASH_2_5",
    "execute": false
  }'
```

## Security

- API key authentication on all endpoints (except `/health`)
- CORS restricted to allowed origins
- Environment variables for all secrets
- No secrets in code or logs

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `baml-py` - BAML client for LLM integration
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation
- `chromadb` - Vector database for embeddings
- `sentence-transformers` - Embeddings model
