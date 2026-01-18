# Mora Sleep Scoring - System Architecture

This document explains how the web-based Mora sleep scoring system works, including all services, components, and the flow of data.

---

## System Overview Diagram

```
                                    MORA SLEEP SCORING SYSTEM
 =====================================================================================================

                                         USER INTERFACES
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                                     │
 │   ┌─────────────────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────┐ │
 │   │     WEB BROWSER             │      │     COMMAND LINE            │      │    PYTHON SCRIPT    │ │
 │   │   (Non-technical users)     │      │   (Power users)             │      │   (Developers)      │ │
 │   │                             │      │                             │      │                     │ │
 │   │  https://sleep-cz3.pages.dev│      │  $ mora score file.wav      │      │  requests.post(...) │ │
 │   └──────────────┬──────────────┘      └──────────────┬──────────────┘      └──────────┬──────────┘ │
 │                  │                                    │                                │            │
 └──────────────────┼────────────────────────────────────┼────────────────────────────────┼────────────┘
                    │                                    │                                │
                    │ HTTPS                              │ HTTPS                          │ HTTPS
                    │                                    │                                │
                    ▼                                    ▼                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                    CLOUD INFRASTRUCTURE                                             │
 │                                                                                                     │
 │   ┌───────────────────────────────────────┐      ┌─────────────────────────────────────────────────┐│
 │   │          CLOUDFLARE PAGES             │      │              RAILWAY                            ││
 │   │         (Static Hosting)              │      │        (Container Hosting)                      ││
 │   │                                       │      │                                                 ││
 │   │  ┌─────────────────────────────────┐  │      │  ┌───────────────────────────────────────────┐  ││
 │   │  │     REACT FRONTEND              │  │      │  │         FASTAPI BACKEND                  │  ││
 │   │  │                                 │  │      │  │                                           │  ││
 │   │  │  • index.html                   │  │      │  │  POST /api/v1/score                       │  ││
 │   │  │  • app.js (compiled TypeScript) │──┼──────┼──│    → Accepts WAV file                     │  ││
 │   │  │  • styles.css                   │  │  API │  │    → Returns sleep scores + signals       │  ││
 │   │  │                                 │  │ calls│  │                                           │  ││
 │   │  │  Components:                    │  │      │  │  GET /api/v1/model/info                   │  ││
 │   │  │  • FileUpload                   │  │      │  │    → Model version, features              │  ││
 │   │  │  • EEGPlot / EMGPlot            │  │      │  │                                           │  ││
 │   │  │  • PowerSpectrum                │  │      │  │  GET /api/v1/health                       │  ││
 │   │  │  • RelativePower                │  │      │  │    → API status check                     │  ││
 │   │  │  • Hypnogram                    │  │      │  │                                           │  ││
 │   │  │  • ScoringResults               │  │      │  └───────────────────────────────────────────┘  ││
 │   │  │  • EpochEditor                  │  │      │                                                 ││
 │   │  │                                 │  │      │  ┌───────────────────────────────────────────┐  ││
 │   │  └─────────────────────────────────┘  │      │  │         XGBOOST MODEL                    │  ││
 │   │                                       │      │  │                                           │  ││
 │   │  Built with:                          │      │  │  • Version: 1.2.4                         │  ││
 │   │  • React 19 + TypeScript              │      │  │  • Classes: Wake, Non REM, REM            │  ││
 │   │  • Vite (build tool)                  │      │  │  • Features: 9 input columns              │  ││
 │   │  • Plotly.js (charts)                 │      │  │  • Loaded at startup                      │  ││
 │   │  • Tailwind CSS                       │      │  └───────────────────────────────────────────┘  ││
 │   │  • Zustand (state)                    │      │                                                 ││
 │   └───────────────────────────────────────┘      └─────────────────────────────────────────────────┘│
 └─────────────────────────────────────────────────────────────────────────────────────────────────────┘

 =====================================================================================================
```

---

## The Big Picture (Simplified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER'S BROWSER                                 │
│                                                                             │
│   React App (JavaScript)                                                    │
│   - Renders UI (upload button, hypnogram, EEG/EMG plots, etc.)             │
│   - Runs entirely in the browser                                            │
│   - Makes HTTP requests to backend                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP Requests
                                    │ (upload WAV, get scores)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAILWAY (Backend)                              │
│                                                                             │
│   FastAPI + Python                                                          │
│   - Receives WAV files                                                      │
│   - Extracts features (power spectra, signal stats)                         │
│   - Runs XGBoost model                                                      │
│   - Returns JSON scores + signal data for visualization                     │
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
| **Charts** | Plotly.js | EEG/EMG time series, spectrogram, hypnogram |
| **State** | Zustand | Global state management |
| **Styling** | Tailwind CSS | Utility-first CSS |
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

## Data Flow: Scoring a Recording

```
 USER                    FRONTEND                   BACKEND                    MODEL
  │                         │                          │                         │
  │  1. Select WAV file     │                          │                         │
  ├────────────────────────►│                          │                         │
  │                         │                          │                         │
  │                         │  2. POST /api/v1/score   │                         │
  │                         │     (multipart/form-data)│                         │
  │                         ├─────────────────────────►│                         │
  │                         │                          │                         │
  │                         │                          │  3. Read WAV file       │
  │                         │                          ├────────────────────────►│
  │                         │                          │                         │
  │                         │                          │  4. Extract features    │
  │                         │                          │     per epoch           │
  │                         │                          │◄────────────────────────┤
  │                         │                          │                         │
  │                         │                          │  5. model.predict()     │
  │                         │                          ├────────────────────────►│
  │                         │                          │                         │
  │                         │                          │  6. Predictions         │
  │                         │                          │◄────────────────────────┤
  │                         │                          │                         │
  │                         │                          │  7. Apply rule filter   │
  │                         │                          │     (clean up errors)   │
  │                         │                          │                         │
  │                         │  8. JSON Response        │                         │
  │                         │     {epochs, summary,    │                         │
  │                         │      eeg_data, emg_data, │                         │
  │                         │      power_spectrum}     │                         │
  │                         │◄─────────────────────────┤                         │
  │                         │                          │                         │
  │  9. Display plots       │                          │                         │
  │     & results           │                          │                         │
  │◄────────────────────────┤                          │                         │
  │                         │                          │                         │
```

---

## Build Time vs Runtime

### Build Time (happens once, when you deploy)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BUILD TIME                                     │
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
│   Static files hosted at sleep-cz3.pages.dev                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BACKEND BUILD (Railway):                                                  │
│                                                                             │
│   Dockerfile                                                                │
│         │                                                                   │
│         ▼  docker build                                                     │
│   Docker Image (Python + FastAPI + Model)                                   │
│         │                                                                   │
│         ▼  Start container                                                  │
│   Server running at sleep-production.up.railway.app                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Runtime (happens every time a user visits)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               RUNTIME                                       │
│                          (user visits the app)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USER OPENS https://sleep-cz3.pages.dev                                  │
│         │                                                                   │
│         ▼                                                                   │
│  2. CLOUDFLARE serves JavaScript/HTML/CSS files                             │
│         │                                                                   │
│         ▼                                                                   │
│  3. BROWSER executes JavaScript (React app starts)                          │
│         │                                                                   │
│         │  (Cloudflare's job is done - it just served files)                │
│         ▼                                                                   │
│  4. USER uploads a WAV file                                                 │
│         │                                                                   │
│         ▼                                                                   │
│  5. BROWSER sends HTTP POST to Railway backend                              │
│         │   POST https://sleep-production.up.railway.app/api/v1/score       │
│         │   Body: WAV file + baseline epoch                                 │
│         ▼                                                                   │
│  6. RAILWAY receives request                                                │
│         │   FastAPI routes to score_file() function                         │
│         │   Python extracts features                                        │
│         │   XGBoost predicts sleep stages                                   │
│         ▼                                                                   │
│  7. RAILWAY returns JSON response                                           │
│         │   {"epochs": [...], "eeg_data": [...], "summary": {...}}          │
│         ▼                                                                   │
│  8. BROWSER receives response                                               │
│         │   React updates UI                                                │
│         │   Plotly renders EEG/EMG/Hypnogram                                │
│         ▼                                                                   │
│  9. USER sees scored data with full visualizations                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Frontend Component Architecture

```
 App.tsx
 │
 ├── Header
 │   └── Title, API status indicator
 │
 ├── FileUpload.tsx
 │   ├── Drag-drop zone for WAV files
 │   ├── Baseline epoch selector
 │   └── Upload progress bar
 │
 ├── EpochViewer.tsx
 │   ├── Navigation Controls
 │   │   ├── Previous/Next epoch buttons
 │   │   ├── Find epoch input
 │   │   ├── Next REM button
 │   │   └── Window size selector (1/3/5/7 epochs)
 │   │
 │   ├── EEGPlot.tsx
 │   │   └── Time series with epoch shading by sleep state
 │   │
 │   ├── PowerSpectrum.tsx
 │   │   └── FFT frequency plot (0-50 Hz)
 │   │
 │   ├── EMGPlot.tsx
 │   │   └── Time series with epoch shading
 │   │
 │   └── RelativePower.tsx
 │       └── Delta/Theta bar chart
 │
 ├── ScoringPanel.tsx
 │   ├── Score buttons (Wake / NREM / REM / Unscored)
 │   ├── Keyboard shortcuts: W / E / R / T
 │   └── Current epoch info display
 │
 ├── Hypnogram.tsx
 │   ├── Full recording timeline
 │   ├── Clickable navigation to any epoch
 │   └── Color-coded by sleep state
 │
 ├── ScoringResults.tsx
 │   ├── Summary statistics (counts, percentages)
 │   └── Pie/bar chart visualization
 │
 └── ExportPanel.tsx
     ├── Export Scores (CSV)
     ├── Export Breakdown (summary CSV)
     └── Import Scores (load existing scores)
```

---

## Feature Engineering Pipeline

```
 RAW WAV FILE                    FEATURE EXTRACTION                    MODEL INPUT
 ─────────────────────────────────────────────────────────────────────────────────

 ┌──────────────────┐
 │ Stereo WAV       │
 │ • Channel 1: EEG │
 │ • Channel 2: EMG │
 │ • 1000 Hz        │
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────┐     ┌────────────────────────────────────────────────────┐
 │ Split into       │     │  For each 10-second epoch:                        │
 │ 10-second epochs │────►│                                                    │
 │ (10,000 samples) │     │  EEG Features:           EMG Features:            │
 └──────────────────┘     │  • quantile_80           • std                    │
                          │  • peak-to-peak          • events (muscle bursts) │
                          │  • sum of squares        • peak-to-peak           │
                          │                                                    │
                          │  Power Features (FFT):                            │
                          │  • delta_rel (0.5-4 Hz relative power)            │
                          │  • theta_rel (5.5-8.5 Hz relative power)          │
                          │  • theta_over_delta ratio                         │
                          └────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────────────────┐
                          │  Normalize to baseline epoch (user-selected Wake) │
                          │  Features = (raw - baseline_mean) / baseline_std   │
                          └────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────────────────┐
                          │  XGBoost Classifier                                │
                          │  Input: 9 features per epoch                       │
                          │  Output: Wake / Non REM / REM                      │
                          └────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────────────────┐
                          │  Rule-Based Filter (post-processing)               │
                          │  • Wake-Wake-REM → Wake-Wake-Wake                  │
                          │  • Isolated single epochs → match neighbors        │
                          │  • Isolated double epochs → match neighbors        │
                          └────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Why separate Frontend and Backend?

| Aspect | Frontend Only | Backend Required |
|--------|---------------|------------------|
| Display UI | Yes | |
| Handle clicks | Yes | |
| Run Python | No | Yes |
| Load ML models | No | Yes |
| Process large files | No | Yes |

Browsers can't run Python or load scikit-learn models. We need a server for that.

### What is an API?

**API = Application Programming Interface**

It's a contract: "Send me data in this format, I'll return results in that format."

```
Request:
  POST /api/v1/score
  Body: WAV file + baseline_epoch

Response:
  {
    "epochs": [{"epoch": 0, "score": "Wake"}, ...],
    "summary": {"Wake": {"count": 120, "percentage": 45.2}, ...},
    "eeg_data": [[...], [...], ...],   // per-epoch signal data
    "emg_data": [[...], [...], ...],
    "power_data": [[...], [...], ...]
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

---

## Security Model

```
┌────────────────────────────────────────────────────────────────────┐
│                        SECURITY LAYERS                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. TRANSPORT SECURITY                                             │
│     • All traffic over HTTPS (TLS 1.3)                            │
│     • Certificates managed by Cloudflare/Railway                   │
│                                                                    │
│  2. API AUTHENTICATION (Optional)                                  │
│     • X-API-Key header                                            │
│     • Keys hashed with SHA-256                                    │
│     • Configure via MORA_API_KEYS env var                         │
│                                                                    │
│  3. RATE LIMITING                                                  │
│     • Per-client IP address                                       │
│     • Configurable requests/minute                                │
│     • 429 response when exceeded                                  │
│                                                                    │
│  4. CORS POLICY                                                    │
│     • Restrict to specific frontend domains                       │
│     • Configure via MORA_CORS_ORIGINS                             │
│                                                                    │
│  5. INPUT VALIDATION                                               │
│     • File type checking (.wav only)                              │
│     • File size limits (500MB max)                                │
│     • Pydantic schema validation                                  │
│                                                                    │
│  6. DATA HANDLING                                                  │
│     • No server-side storage of recordings                        │
│     • Process in memory, return results, discard                  │
│     • No user accounts or PII stored                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
 GIT REPOSITORY (GitHub)
 ────────────────────────────────────────────────────────────────────
         │
         │ push to main/claude/refactor
         │
         ├─────────────────────────────────┬──────────────────────────
         │                                 │
         ▼                                 ▼
 ┌─────────────────────────┐      ┌─────────────────────────┐
 │   CLOUDFLARE PAGES      │      │       RAILWAY           │
 │                         │      │                         │
 │ Watches: /frontend      │      │ Watches: /backend       │
 │                         │      │                         │
 │ Build command:          │      │ Build: Dockerfile       │
 │   npm run build         │      │                         │
 │                         │      │ Start: start.sh         │
 │ Output: /frontend/dist  │      │   uvicorn app.main:app  │
 │                         │      │   --host 0.0.0.0        │
 │ URL:                    │      │   --port $PORT          │
 │ sleep-cz3.pages.dev     │      │                         │
 │                         │      │ URL:                    │
 │ Environment:            │      │ sleep-production.       │
 │ VITE_API_URL=https://   │      │   up.railway.app        │
 │   sleep-production...   │      │                         │
 └─────────────────────────┘      │ Environment:            │
                                  │ MORA_API_KEYS=...       │
                                  │ MORA_RATE_LIMIT_...     │
                                  └─────────────────────────┘
```

---

## URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | https://sleep-cz3.pages.dev | Web interface |
| Backend | https://sleep-production.up.railway.app | API server |
| API Docs | https://sleep-production.up.railway.app/docs | Swagger UI |
| Health Check | https://sleep-production.up.railway.app/api/v1/health | Status |

---

## Cost

| Service | Role | Cost |
|---------|------|------|
| Cloudflare Pages | Serve frontend files | Free |
| Railway | Run Python backend | ~$5-15/month |
| **Total** | | **~$5-15/month** |

---

## Local Development

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000, proxies API to :8000
```

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
9. **Railway processes** file, returns scores + signal data
10. **Browser displays** full visualization (EEG, EMG, spectrum, hypnogram)

The frontend and backend are completely separate services that communicate over HTTP.
