# Mora Sleep Scoring - Refactor Review

## Phase 1: Backend API - COMPLETED

### Summary
Implemented a FastAPI backend that replicates the sleep scoring pipeline from the PyQt5 GUI, exposing it as a REST API. This enables non-technical users to score sleep data via a web interface (Phase 2) and power users to access it via CLI or programmatically.

### Files Created

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI application entry point with CORS and lifespan handling |
| `backend/app/core/config.py` | Pydantic settings (model path, feature columns, upload limits) |
| `backend/app/core/model_loader.py` | Model loading and prediction service |
| `backend/app/ml/sleep_functions.py` | Feature engineering (migrated from `data_processing/`) |
| `backend/app/services/scoring_service.py` | Full scoring pipeline orchestration |
| `backend/app/schemas/scoring.py` | Pydantic request/response models |
| `backend/app/api/v1/endpoints/score.py` | POST /score and /features endpoints |
| `backend/app/api/v1/endpoints/model.py` | GET /model/info and /health endpoints |
| `backend/app/api/v1/router.py` | API router aggregation |
| `backend/requirements.txt` | Python dependencies |
| `backend/Dockerfile` | Container image definition |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/score` | POST | Score a WAV file, returns epoch scores + summary |
| `/api/v1/features` | POST | Extract features only (for debugging) |
| `/api/v1/model/info` | GET | Model metadata (version, features, classes) |
| `/api/v1/health` | GET | Health check (model loaded status) |
| `/docs` | GET | Interactive Swagger documentation |

### Validation Results

**Server Startup:**
- Model loads successfully: XGBoost v1.2.4
- Health endpoint returns `{"status":"healthy","model_loaded":true}`

**Scoring Test (`sample_data/sample_signal.wav`):**
- 544 epochs scored (1.51 hours of recording)
- Samplerate: 1000 Hz
- Processing time: ~2 seconds for 21MB file
- Results consistent with direct model invocation

**API Response Format:**
```json
{
  "success": true,
  "epochs": [{"epoch": 0, "score": "Wake", "timestamp_seconds": 0.0}, ...],
  "summary": {
    "total_epochs": 544,
    "wake_count": 294, "wake_percent": 54.04,
    "nrem_count": 181, "nrem_percent": 33.27,
    "rem_count": 69, "rem_percent": 12.68,
    "recording_duration_hours": 1.51
  },
  "model_version": "1.2.4",
  "samplerate": 1000,
  "baseline_epoch": 2
}
```

### Known Issues / Limitations

1. **XGBoost Version Warning**: Model was saved with older XGBoost version. Consider re-saving with `Booster.save_model()` for better compatibility.

2. **sklearn Version Warning**: LabelEncoder was pickled with sklearn 1.4.1, running on 1.6.1. Works but may cause issues in edge cases.

3. **Python 3.9 Compatibility**: All type hints updated to use `typing` module (List, Dict, Optional, Tuple) instead of Python 3.10+ syntax.

4. **Async Processing Not Implemented**: Large file (>50MB) async job queue from Phase 1 plan not yet implemented. Files are processed synchronously.

### Dependencies Installed

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pydantic>=2.5.0
pydantic-settings>=2.1.0
xgboost>=2.0.0
scikit-learn>=1.4.0
scipy>=1.11.0
numpy>=1.26.0
pandas>=2.0.0
joblib>=1.3.0
```

---

## Running the Backend

### Local Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then visit: http://localhost:8000/docs

### Test with curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Model info
curl http://localhost:8000/api/v1/model/info

# Score a file
curl -X POST -F "file=@sample_data/sample_signal.wav" -F "start_epoch=2" \
  http://localhost:8000/api/v1/score
```

---

## Phase 2: Web Frontend - COMPLETED

### Summary
Created a React + TypeScript web application with drag-and-drop file upload, interactive hypnogram visualization using Plotly.js, and CSV export functionality. The frontend communicates with the backend API and provides a user-friendly interface for non-technical users.

### Technology Stack
- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **Charts**: Plotly.js (react-plotly.js)
- **State Management**: Zustand
- **File Upload**: react-dropzone

### Files Created

| File | Purpose |
|------|---------|
| `frontend/src/App.tsx` | Main application component with layout |
| `frontend/src/api/sleepApi.ts` | API client for backend communication |
| `frontend/src/store/useAppStore.ts` | Global state management (Zustand) |
| `frontend/src/types/scoring.ts` | TypeScript type definitions |
| `frontend/src/components/FileUpload.tsx` | Drag-and-drop file upload with progress |
| `frontend/src/components/Hypnogram.tsx` | Interactive sleep stage visualization |
| `frontend/src/components/ScoringResults.tsx` | Summary statistics display |
| `frontend/src/components/ExportButton.tsx` | CSV export functionality |
| `frontend/vite.config.ts` | Vite config with Tailwind and API proxy |
| `frontend/src/index.css` | Tailwind imports and custom styles |

### Features

1. **Drag-and-drop file upload** with progress indicator
2. **Baseline epoch selector** for normalization
3. **Interactive hypnogram** showing sleep states over time
4. **Summary statistics** with color-coded breakdown
5. **Proportion bar** visualizing Wake/NREM/REM percentages
6. **CSV export** of epoch-by-epoch scores
7. **API health status** indicator
8. **Responsive design** for various screen sizes

### Validation Results

- Frontend builds successfully (`npm run build`)
- Vite dev server runs on port 3000
- API proxy correctly forwards `/api/*` requests to backend
- All TypeScript type checks pass

### Dependencies Installed

```json
{
  "dependencies": {
    "react": "^18.x",
    "react-dom": "^18.x",
    "react-dropzone": "^14.x",
    "react-plotly.js": "^2.x",
    "plotly.js": "^2.x",
    "zustand": "^4.x"
  },
  "devDependencies": {
    "typescript": "^5.x",
    "vite": "^7.x",
    "tailwindcss": "^4.x",
    "@tailwindcss/vite": "^4.x",
    "@types/react-plotly.js": "^2.x"
  }
}
```

---

## Running the Full Stack

### Start Both Servers

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000 in your browser.

### Production Build

```bash
cd frontend
npm run build
# Static files in frontend/dist/
```

---

## Next Steps

| Phase | Status | Priority |
|-------|--------|----------|
| Phase 1: Backend API | **COMPLETED** | High |
| Phase 2: Web Frontend | **COMPLETED** | High |
| Phase 3: Model Diagnostics | Not started | Medium |
| Phase 4: Testing | Not started | Medium |
| Phase 5: CLI Tool | Not started | Low |
| Phase 6: Deployment | Not started | Medium |
| Phase 7: Security | Not started | Medium |
| Phase 8: PyQt5 Deprecation | Not started | Low |

### Recommended Next Actions

1. **Phase 6: Deployment** - Deploy to Railway (backend) and Cloudflare Pages (frontend)
2. **Phase 3: Model Diagnostics** - Add cross-validation, Cohen's Kappa, transition analysis
3. **Fix model serialization** - Re-save XGBoost model to eliminate version warnings
