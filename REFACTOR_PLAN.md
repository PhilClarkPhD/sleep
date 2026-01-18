# Mora Sleep Scoring Project Improvement Plan

## Current Status (Updated: January 2025)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Backend API | ✅ **COMPLETED** | FastAPI backend fully functional |
| Phase 2: Web Frontend | ✅ **COMPLETED** | React app with Plotly hypnogram |
| Phase 3: Model Validation | ✅ **COMPLETED** | Time-series CV, diagnostics module |
| Phase 4: Testing | ✅ **COMPLETED** | Unit + integration tests |
| Phase 5: CLI Tool | ✅ **COMPLETED** | `mora score` command |
| Phase 6: Deployment | ✅ **COMPLETED** | Railway + Cloudflare Pages live |
| Phase 7: Security | ✅ **COMPLETED** | API keys, rate limiting, logging |
| Phase 8: PyQt5 Deprecation | 🔄 **IN PROGRESS** | Frontend feature parity achieved |

**Branch:** `claude/refactor`

**Live URLs:**
- Frontend: https://sleep-cz3.pages.dev
- Backend: https://sleep-production.up.railway.app
- API Docs: https://sleep-production.up.railway.app/docs

**Next step:** Deploy latest changes, then finalize PyQt5 deprecation

---

## Executive Summary

Transform the Mora sleep scoring application from a Python-dependent PyQt5 desktop app to a modern, accessible web-based system with REST API, while improving model validation and diagnostics.

## Requirements Addressed

| Requirement | Current State | Proposed Solution |
|-------------|---------------|-------------------|
| 1. UI for loading, visualizing, scoring, exporting | PyQt5 desktop (requires Python) | Web-based React UI |
| 2. Model that featurizes and scores | XGBoost works but limited validation | Add comprehensive diagnostics & validation |
| 3. Model callable from UI + CLI API | No API, Python-only | FastAPI REST API + CLI tool |
| 4. Accessible for non-technical users | Requires Python env | Browser-based, no installation needed |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Access                             │
├─────────────────────────────────────────────────────────────────┤
│   Browser (Non-technical)          CLI (Power users)            │
│   https://mora.example.com         mora-cli score file.wav      │
└──────────────┬───────────────────────────┬──────────────────────┘
               │                           │
               ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Railway)                    │
│   POST /api/v1/score     - Score WAV file                       │
│   POST /api/v1/features  - Extract features only                │
│   GET  /api/v1/model/info - Model metadata                      │
│   GET  /api/v1/health    - Health check                         │
├─────────────────────────────────────────────────────────────────┤
│   XGBoost Model (embedded) + Feature Engineering + Rule Filter  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Backend API ✅ COMPLETED

### Create FastAPI backend
**Location:** `/Users/phil/philclarkphd/sleep/backend/`
**Status:** Fully implemented and tested

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── score.py        # POST /score
│   │   │   └── model.py        # GET /model/info
│   │   └── schemas/            # Pydantic models
│   ├── core/
│   │   └── model_loader.py     # Load XGBoost at startup
│   ├── services/
│   │   └── scoring_service.py  # Scoring pipeline
│   └── ml/
│       ├── sleep_functions.py  # Migrated from data_processing/
│       └── rule_filter.py      # Rule-based filtering
├── models/
│   └── xgboost_v1.2.4.pkl
├── Dockerfile
└── requirements.txt
```

### Key files to migrate:
- `data_processing/sleep_functions.py` → `backend/app/ml/sleep_functions.py`
  - `compute_power()`, `smooth_signal()`, `compute_relative_power()`
  - `generate_features()`, `apply_rule_based_filter()`

### API Endpoints:
```
POST /api/v1/score
  Input: WAV file (multipart/form-data) + start_epoch (int)
  Output: { epochs: [...], scores: [...], summary: {...} }

GET /api/v1/model/info
  Output: { version, features, performance_metrics }

GET /api/v1/jobs/{job_id}
  Output: { status: "pending"|"processing"|"complete"|"failed", progress: 0-100, result?: {...} }
```

### Large File Handling:
- **File size limit:** 500MB max per upload
- **Async processing:** Files > 50MB processed via job queue
  - Return `202 Accepted` with `job_id`
  - Client polls `GET /api/v1/jobs/{job_id}` for status
- **WebSocket option:** `WS /api/v1/jobs/{job_id}/stream` for real-time progress
- **Server timeout:** Configure uvicorn timeout to 300s (default 30s insufficient)
- **Chunked uploads:** Consider `tus.io` protocol for resumable uploads

---

## Phase 2: Web Frontend ✅ COMPLETED

### Create React frontend
**Location:** `/Users/phil/philclarkphd/sleep/frontend/`
**Status:** Fully implemented and tested locally

```
frontend/
├── src/
│   ├── components/
│   │   ├── FileUpload.tsx      # Drag-and-drop WAV upload
│   │   ├── Hypnogram.tsx       # Sleep stage visualization
│   │   ├── EEGPlot.tsx         # EEG signal plot (Plotly.js)
│   │   ├── EMGPlot.tsx         # EMG signal plot
│   │   └── ExportButton.tsx    # CSV export
│   ├── api/
│   │   └── sleepApi.ts         # API client
│   └── pages/
│       └── HomePage.tsx        # Main scoring interface
├── package.json
└── vite.config.ts
```

### Technology stack:
- **Framework:** React 18 + TypeScript + Vite
- **Charts:** Plotly.js (handles large time series, WebGL)
- **Styling:** Tailwind CSS
- **File upload:** react-dropzone

### EEG/EMG Visualization Strategy:
Large EEG recordings can have millions of data points. Rendering all at once causes browser performance issues.

- **Viewport windowing:** Only render visible time range (e.g., 60s window)
- **Server-side downsampling:** `GET /api/v1/overview?resolution=1000` returns decimated signal for overview plot
- **Progressive loading:** On zoom, fetch higher-resolution data for visible region
- **WebGL rendering:** Plotly.js with `scattergl` trace type for hardware acceleration
- **Virtual scrolling:** For hypnogram epoch list, use `react-window` or `tanstack-virtual`

### Frontend State Management:
- **Recommended:** [Zustand](https://github.com/pmndrs/zustand) - lightweight, minimal boilerplate
- **State slices:**
  ```typescript
  interface AppState {
    // File upload
    uploadState: 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
    uploadProgress: number;
    jobId: string | null;

    // Scoring results
    epochs: Epoch[];
    scores: Score[];
    summary: SummaryStats | null;

    // UI settings
    viewportStart: number;
    viewportEnd: number;
    selectedEpoch: number | null;
    showEMG: boolean;
  }
  ```
- **Alternative:** React Context + useReducer for simpler apps (no external dependency)

---

## Phase 3: Model Validation & Diagnostics (Priority: Medium)

### 3.1 Fix cross-validation
**Problem:** Current `RandomizedSearchCV` uses standard k-fold which causes data leakage in time-series data.

**Solution:** Create time-series aware CV in `/Users/phil/philclarkphd/sleep/model/cross_validation.py`:
- Group-aware time-series split (respects ID_day grouping)
- Add gap between train/test to prevent leakage from adjacent epochs

**Modify:** `model/train_model.py` to accept custom CV splitter

### 3.2 Add diagnostics module
**Create:** `/Users/phil/philclarkphd/sleep/model/diagnostics.py`

Key metrics to add:
- **Cohen's Kappa** (standard in sleep scoring literature)
- **Balanced accuracy** (handles class imbalance)
- **Per-class metrics** (especially REM recall)
- **Transition analysis** (errors at state transitions)
- **Calibration curves** (probability reliability)

### 3.3 Add monitoring
**Create:** `/Users/phil/philclarkphd/sleep/model/monitoring.py`
- Detect feature distribution drift (KS test)
- Detect label distribution shift
- Performance degradation alerts

---

## Phase 4: Testing (Priority: Medium)

### Test structure:
```
tests/
├── conftest.py                    # Fixtures
├── unit/
│   ├── test_feature_extraction.py
│   ├── test_train_model.py
│   └── test_rule_based_filter.py  # Existing, move here
├── integration/
│   └── test_training_pipeline.py
└── regression/
    ├── test_model_performance.py
    └── baseline_metrics.json
```

### Key tests to add:
1. Feature extraction output validation (no NaN/inf)
2. Train/test split data leakage checks
3. Model serialization round-trip
4. Regression tests against baseline metrics

---

## Phase 5: CLI Tool (Priority: Low)

**Create:** `/Users/phil/philclarkphd/sleep/cli/score_cli.py`

```bash
# Install
pip install mora-cli

# Usage
mora-cli score recording.wav --output scores.csv
mora-cli score recording.wav --api-url https://mora.example.com
```

---

## Phase 6: Deployment 🔄 IN PROGRESS

**See [DEPLOYMENT.md](./DEPLOYMENT.md) for step-by-step deployment instructions.**

### Files Created for Deployment:
- `backend/Dockerfile` - Python 3.11-slim image with FastAPI
- `backend/railway.json` - Railway deployment configuration
- `docker-compose.yml` - Local Docker development setup
- `frontend/Dockerfile` - Multi-stage build with nginx
- `frontend/nginx.conf` - Nginx configuration for SPA routing

### Recommended: Railway
- **Cost:** ~$7-15/month
- **Backend:** Docker container running FastAPI
- **Frontend:** Static files (or separate Vercel/Cloudflare)

### Dockerfile:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Local Deployment Option:
For labs with sensitive data or air-gapped environments:

**Docker Compose (local-only):**
```yaml
# docker-compose.local.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models  # Mount local model artifacts
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://localhost:8000
```

**Quick start for air-gapped environments:**
```bash
# Build images (requires internet once)
docker compose -f docker-compose.local.yml build

# Run locally (no internet required)
docker compose -f docker-compose.local.yml up

# Access at http://localhost:3000
```

**Frontend API URL configuration:**
- Development: `VITE_API_URL=http://localhost:8000`
- Cloud: `VITE_API_URL=https://mora-api.railway.app`
- Configurable at build time via environment variable

---

## Phase 7: Security & Data Handling (Priority: Medium)

### HTTPS Enforcement:
- Railway/Vercel provide automatic HTTPS
- For local Docker: use reverse proxy (Caddy/nginx) with self-signed cert for internal networks
- Backend should redirect HTTP → HTTPS in production

### Authentication:
- **Simple option:** API key in header (`X-API-Key: xxx`)
  - Generate per-lab keys, store hashed in database
  - Rate limit by key (e.g., 100 requests/hour)
- **Advanced option:** OAuth2 with institutional SSO (for multi-institution deployments)

### Data Retention Policy:
- **Default:** Auto-delete uploaded files after scoring completes (no server-side storage)
- **Optional:** Retain for N days if user opts in (for debugging/reprocessing)
- **Job results:** Keep metadata for 7 days, then purge
- **Document clearly:** "Your data is processed and immediately deleted. We do not store recordings."

### IRB/Research Data Compliance:
- For HIPAA/IRB-sensitive data: recommend self-hosted option
- Document that cloud deployment processes data but doesn't retain
- Provide data processing agreement (DPA) template for institutional review
- Consider EU GDPR if international users: add data location disclosure

### Self-Hosting for Sensitive Data:
- Provide `docker-compose.local.yml` (see Phase 6)
- Document firewall/VPN setup for lab-only access
- No telemetry or analytics in self-hosted mode

---

## Phase 8: PyQt5 App Deprecation (Priority: Low)

### Deprecation Timeline:
| Milestone | Action |
|-----------|--------|
| Web app launch | Announce deprecation, add banner to PyQt5 app |
| Launch + 3 months | Stop adding features to PyQt5, security fixes only |
| Launch + 6 months | Archive repository, final release |
| Launch + 12 months | Remove from documentation |

### Migration Guide for Existing Users:
1. **Data compatibility:** Ensure CSV export format matches between PyQt5 and web app
2. **Feature parity checklist:**
   - [x] WAV file loading
   - [x] Manual epoch editing (keyboard shortcuts: W/E/R/T)
   - [x] Hypnogram visualization (clickable navigation)
   - [x] CSV export (scores + breakdown)
   - [x] CSV import (restore previous work)
   - [x] EEG/EMG time series plots
   - [x] Power spectrum with delta/theta bands
   - [x] Relative power bar chart
   - [x] Epoch navigation (arrow keys + window size)
   - [ ] Batch processing (CLI supports via `mora batch`)
3. **Documentation:** Create "Migrating from Desktop to Web App" guide

### Archive Strategy:
- Tag final release as `v2.0-final-desktop`
- Move to `mora-desktop-archived` repository
- Keep PyPI package available but mark as deprecated
- README points to web app as replacement

---

## Critical Files

| Purpose | Current Location | Action |
|---------|------------------|--------|
| Feature engineering | `data_processing/sleep_functions.py` | Migrate to backend |
| Scoring pipeline | `mora/update_objects.py:539-578` | Replicate in API |
| Model config | `model/model_config.json` | Use for API config |
| Model artifacts | `model_artifacts/XGBoost_1.2.4/` | Copy to backend |
| Training | `model/train_model.py` | Add CV + diagnostics |

---

## Verification Plan

1. **Backend API:**
   - `curl -X POST /api/v1/score -F "file=@sample_data/sample_signal.wav"`
   - Compare output against Mora GUI scoring

2. **Frontend:**
   - Upload sample_data/sample_signal.wav
   - Verify hypnogram matches expected scores
   - Export CSV and compare to manual scores

3. **Model validation:**
   - Run new cross-validation and compare F1 scores
   - Generate model card with diagnostics

4. **Tests:**
   - `pytest tests/ -v`
   - Check baseline regression tests pass

---

## Implementation Order

1. **Backend API** (highest impact on accessibility)
2. **Web Frontend** (completes user-facing requirements)
3. **Model Diagnostics** (improves scientific rigor)
4. **Testing** (ensures reliability)
5. **Deployment** (makes it publicly accessible)
6. **CLI Tool** (power user feature)
7. **Security & Data Handling** (required before public launch)
8. **PyQt5 Deprecation** (post-launch cleanup)

---

## Estimated Ongoing Costs

| Service | Monthly Cost |
|---------|--------------|
| Railway (backend) | $7-15 |
| Cloudflare Pages (frontend) | Free |
| **Total** | **~$10/month** |

---

## Documentation Requirements

### REFACTOR_REVIEW.md ✅ Created
See `/Users/phil/philclarkphd/sleep/REFACTOR_REVIEW.md` for:
- Summary of changes made per phase
- Files created/modified/deleted
- Validation results (tests passed, manual verification)
- Known issues or limitations
- Next steps or follow-up items
- Instructions for running the new system

---

## Implementation Insights & Lessons Learned

These notes capture lessons learned during Phase 1 & 2 implementation to help future development.

### Python Version Compatibility
- **Issue:** Python 3.9 doesn't support `X | Y` union syntax (PEP 604)
- **Solution:** Use `from typing import Optional, List, Dict, Tuple` and `Optional[X]` syntax
- **Affected files:** All backend Python files use typing module for compatibility

### Environment Setup
- **Node.js required:** Install with `brew install node` on macOS
- **Python deps:** `pip install -r backend/requirements.txt`
- **Frontend deps:** `cd frontend && npm install`

### Running Locally (Quick Start)
```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

### Model Path Configuration
The backend looks for the model in two locations:
1. `backend/models/XGBoost_1.2.4.pkl` (Docker/production)
2. `model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl` (local development)

The model file is ~7MB and committed to the repo for deployment.

### Frontend API Configuration
The frontend uses `VITE_API_URL` environment variable:
- Local dev with Vite proxy: leave unset (Vite handles `/api/` routing)
- Local dev direct: `VITE_API_URL=http://localhost:8000`
- Production: `VITE_API_URL=https://your-railway-url.up.railway.app`

### TypeScript Notes
- Plotly.js title must be `{ text: 'string' }` not just `'string'`
- Unused variables cause build errors (remove or prefix with `_`)

### Verification Test
```bash
# Test backend health
curl http://localhost:8000/api/v1/health
# Expected: {"status":"healthy","model_loaded":true,"version":"1.0.0"}

# Test scoring with sample file
curl -X POST http://localhost:8000/api/v1/score \
  -F "file=@sample_data/sample_signal.wav"
```

### Known Limitations (Current Implementation)
1. ~~No authentication/rate limiting~~ ✅ Implemented in Phase 7 (optional, disabled by default)
2. No async job queue for large files (files process synchronously)
3. ~~No EEG/EMG signal plots in frontend~~ ✅ Added in Phase 8 (SignalPlot, PowerSpectrum, RelativePower)
4. ~~No manual epoch editing in frontend~~ ✅ Added in Phase 8 (keyboard shortcuts W/E/R/T)
5. No batch processing in web UI (CLI supports batch via `mora batch`)
