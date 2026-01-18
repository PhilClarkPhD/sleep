# Mora Sleep Scoring - Architecture Overview

This document explains how the web-based Mora sleep scoring system works, including all services, components, and the flow of data.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER'S BROWSER                                  │
│                                                                             │
│   React App (JavaScript)                                                    │
│   - Renders UI (upload button, hypnogram, etc.)                            │
│   - Runs entirely in the browser                                           │
│   - Makes HTTP requests to backend                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP Requests
                                      │ (upload WAV, get scores)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAILWAY (Backend)                               │
│                                                                             │
│   FastAPI + Python                                                          │
│   - Receives WAV files                                                      │
│   - Extracts features (power spectra, signal stats)                        │
│   - Runs XGBoost model                                                      │
│   - Returns JSON scores                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components Explained

### Frontend (What the user sees)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | TypeScript | JavaScript with type safety |
| **Library** | React | Build UI components, handle state |
| **Build Tool** | Vite | Compile TypeScript → JavaScript |
| **Hosting** | Cloudflare Pages | Serve static files globally |

The frontend is just files (HTML, JavaScript, CSS). Cloudflare Pages hosts these files and delivers them to users. Once downloaded, the code runs **in the user's browser** - Cloudflare does nothing else.

### Backend (Where the work happens)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python | Feature extraction, ML inference |
| **Framework** | FastAPI | Turn Python functions into HTTP endpoints |
| **Model** | XGBoost | Classify sleep stages |
| **Packaging** | Docker | Bundle code + dependencies |
| **Hosting** | Railway | Run the Docker container 24/7 |

The backend is a Python server. Railway runs it continuously, waiting for requests. When a request arrives, FastAPI routes it to the appropriate function.

---

## Build Time vs Runtime

### Build Time (happens once, when you deploy)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BUILD TIME                                      │
│                         (when you push to GitHub)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FRONTEND BUILD (Cloudflare Pages):                                        │
│                                                                             │
│   TypeScript (.tsx files)                                                   │
│         │                                                                   │
│         ▼  npm run build                                                    │
│   JavaScript + HTML + CSS                                                   │
│         │                                                                   │
│         ▼  Upload to Cloudflare                                             │
│   Static files hosted at sleep-cz3.pages.dev                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BACKEND BUILD (Railway):                                                  │
│                                                                             │
│   Dockerfile                                                                │
│         │                                                                   │
│         ▼  docker build                                                     │
│   Docker Image (Python + FastAPI + Model)                                  │
│         │                                                                   │
│         ▼  Start container                                                  │
│   Server running at sleep-production.up.railway.app                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Runtime (happens every time a user visits)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               RUNTIME                                        │
│                          (user visits the app)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USER OPENS https://sleep-cz3.pages.dev                                 │
│         │                                                                   │
│         ▼                                                                   │
│  2. CLOUDFLARE serves JavaScript/HTML/CSS files                            │
│         │                                                                   │
│         ▼                                                                   │
│  3. BROWSER executes JavaScript (React app starts)                         │
│         │                                                                   │
│         │  (Cloudflare's job is done - it just served files)               │
│         ▼                                                                   │
│  4. USER uploads a WAV file                                                │
│         │                                                                   │
│         ▼                                                                   │
│  5. BROWSER sends HTTP POST to Railway backend                             │
│         │   POST https://sleep-production.up.railway.app/api/v1/score      │
│         │   Body: WAV file                                                  │
│         ▼                                                                   │
│  6. RAILWAY receives request                                               │
│         │   FastAPI routes to score_file() function                        │
│         │   Python extracts features                                        │
│         │   XGBoost predicts sleep stages                                  │
│         ▼                                                                   │
│  7. RAILWAY returns JSON response                                          │
│         │   {"scores": ["Wake", "Wake", "NREM", ...], "summary": {...}}    │
│         ▼                                                                   │
│  8. BROWSER receives response                                              │
│         │   React updates UI                                                │
│         │   Hypnogram renders                                               │
│         ▼                                                                   │
│  9. USER sees scored data                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Why separate Frontend and Backend?

| Aspect | Frontend Only | Backend Required |
|--------|---------------|------------------|
| Display UI | ✅ | |
| Handle clicks | ✅ | |
| Run Python | ❌ | ✅ |
| Load ML models | ❌ | ✅ |
| Process large files | ❌ | ✅ |

Browsers can't run Python or load scikit-learn models. We need a server for that.

### What is an API?

**API = Application Programming Interface**

It's a contract: "Send me data in this format, I'll return results in that format."

```
Request:
  POST /api/v1/score
  Body: WAV file

Response:
  {
    "scores": ["Wake", "NREM", "NREM", "REM", ...],
    "summary": {"Wake": 45.2, "NREM": 38.1, "REM": 16.7}
  }
```

The frontend doesn't know or care how scoring works. It just sends a file and receives JSON.

### What is Docker?

Docker packages your code + all dependencies into a container that runs identically everywhere.

```
Without Docker:
  "Works on my machine" → Fails on server (wrong Python version, missing scipy, etc.)

With Docker:
  Dockerfile defines exact environment → Same result everywhere
```

### What does each service cost?

| Service | Role | Cost |
|---------|------|------|
| Cloudflare Pages | Serve frontend files | Free |
| Railway | Run Python backend | ~$5-15/month |
| **Total** | | **~$5-15/month** |

---

## URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | https://sleep-cz3.pages.dev | User interface |
| Backend | https://sleep-production.up.railway.app | API server |
| Health Check | https://sleep-production.up.railway.app/api/v1/health | Verify backend is running |
| API Docs | https://sleep-production.up.railway.app/docs | Interactive API documentation |

---

## Flow Summary

1. **You write** TypeScript (frontend) + Python (backend)
2. **You push** to GitHub
3. **Cloudflare builds** TypeScript → JavaScript, hosts the files
4. **Railway builds** Docker image, runs the container
5. **User visits** the Cloudflare URL
6. **Browser downloads** JavaScript, runs React app
7. **User uploads** WAV file
8. **Browser sends** request to Railway
9. **Railway processes** file, returns scores
10. **Browser displays** results

The frontend and backend are completely separate services that communicate over HTTP.
