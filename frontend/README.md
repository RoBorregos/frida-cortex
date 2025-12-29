# FRIDA Frontend

Next.js frontend for the FRIDA Command Interpreter web application.

## Features

- Modern UI with gradient design
- Command input with syntax highlighting
- Model selection dropdown
- Real-time command interpretation
- Random command generation
- Execution results visualization
- Responsive mobile-friendly design
- Secure API key handling via Next.js API Routes

## Quick Start

```bash
# Install dependencies
npm install

# Set up environment
cp .env.local.example .env.local
# Edit .env.local with your keys

# Run development server
npm run dev
```

Visit `http://localhost:3000`

## Project Structure

```
src/
├── app/
│   ├── api/                # Next.js API Routes (secure proxy)
│   │   ├── interpret/
│   │   │   └── route.ts    # POST /api/interpret
│   │   ├── generate/
│   │   │   └── route.ts    # POST /api/generate
│   │   └── models/
│   │       └── route.ts    # GET /api/models
│   ├── page.tsx            # Main UI component
│   ├── page.module.css     # Styles
│   ├── layout.tsx          # Root layout
│   └── globals.css         # Global styles
```

## API Routes

All API routes run server-side only and act as secure proxies to the backend.

### `/api/models`
- Method: GET
- Fetches available models from backend
- Adds Authorization header with BACKEND_API_KEY

### `/api/interpret`
- Method: POST
- Body: `{command, model, execute}`
- Interprets natural language command
- Proxies to backend with authentication

### `/api/generate`
- Method: POST
- Body: `{model, execute}`
- Generates random command
- Proxies to backend with authentication

## Environment Variables

```env
# Public (accessible in browser)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8080

# Private (server-side only)
BACKEND_API_KEY=<your-secure-key>
```

**Important**: Only variables prefixed with `NEXT_PUBLIC_` are accessible in the browser. API keys are NOT prefixed and are only available server-side.

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Run production build
npm start

# Lint code
npm run lint
```

## Security

1. **API Routes Proxy Pattern**
   - All backend requests go through Next.js API Routes
   - API keys never exposed to client

2. **Environment Variable Separation**
   - Public vars: `NEXT_PUBLIC_*` (safe for browser)
   - Private vars: No prefix (server-side only)

3. **HTTPS Only**
   - Enforced by Vercel in production
   - Automatic SSL certificates

## Keyboard Shortcuts

- **Ctrl+Enter**: Interpret current command
- **Escape**: Clear results

## Troubleshooting

### "Failed to fetch models"
- Check that backend is running
- Verify NEXT_PUBLIC_BACKEND_URL is correct
- Check browser console for CORS errors

### API key errors
- Ensure BACKEND_API_KEY is set in .env.local
- Verify key matches backend configuration
- Restart dev server after changing env vars

### Styling issues
- Clear browser cache
- Check CSS modules are importing correctly
- Verify globals.css is loaded in layout.tsx
