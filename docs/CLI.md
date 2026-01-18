# Mora CLI Guide

Command-line interface for scoring EEG/EMG recordings.

---

## Installation

```bash
# From the project root
pip install -e ./cli

# Verify installation
mora --version
```

---

## Quick Start

```bash
# Score a file locally (requires model)
mora score recording.wav

# Score using the remote API
mora score recording.wav --api

# Export results to CSV
mora score recording.wav -o scores.csv
```

---

## Commands

### `mora score`

Score a single WAV file.

```bash
mora score FILE [OPTIONS]
```

**Arguments:**
- `FILE` - Path to stereo WAV file (EEG on channel 1, EMG on channel 2)

**Options:**
| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Export scores to CSV file |
| `--api` | Use remote API instead of local model |
| `--api-url URL` | API URL (default: https://sleep-production.up.railway.app) |
| `--start-epoch N` | Baseline epoch for normalization (default: 9) |
| `--json` | Output as JSON instead of table |

**Examples:**

```bash
# Basic local scoring
mora score /path/to/recording.wav

# Score with API and export
mora score recording.wav --api -o results.csv

# Use custom API endpoint
mora score recording.wav --api --api-url http://localhost:8000

# JSON output for scripting
mora score recording.wav --json > results.json
```

---

### `mora batch`

Score all WAV files in a folder.

```bash
mora batch FOLDER -o OUTPUT_FOLDER [OPTIONS]
```

**Arguments:**
- `FOLDER` - Path to folder containing WAV files
- `-o, --output` - Output folder for CSV files (required)

**Options:**
| Option | Description |
|--------|-------------|
| `--api` | Use remote API |
| `--api-url URL` | API URL |
| `--start-epoch N` | Baseline epoch |

**Examples:**

```bash
# Batch score locally
mora batch /path/to/recordings/ -o /path/to/output/

# Batch score via API
mora batch recordings/ -o output/ --api
```

---

### `mora info`

Show model information.

```bash
mora info [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--api-url URL` | API URL to query |

**Example:**

```bash
mora info
# Output:
# Model Information:
#   Name:    XGBoost
#   Version: 1.2.4
#   Classes: Non REM, REM, Wake
#   Features: 9 columns
```

---

## Output Format

### Table Output (default)

```
Summary:
----------------------------------------
State        Epochs       Time        %
----------------------------------------
Wake            180      30.0m     20.8%
Non REM         540      90.0m     62.5%
REM             144      24.0m     16.7%
----------------------------------------
```

### CSV Output (`-o scores.csv`)

```csv
epoch,score
0,Wake
1,Wake
2,Non REM
3,Non REM
...
```

### JSON Output (`--json`)

```json
{
  "scores": ["Wake", "Wake", "Non REM", ...],
  "summary": {
    "Wake": {"count": 180, "percentage": 20.8, "minutes": 30.0},
    "Non REM": {"count": 540, "percentage": 62.5, "minutes": 90.0},
    "REM": {"count": 144, "percentage": 16.7, "minutes": 24.0}
  }
}
```

---

## Local vs API Scoring

| Aspect | Local | API |
|--------|-------|-----|
| Speed | Faster for small files | Faster for large files (server has more RAM) |
| Requirements | Model file + dependencies | Just internet + `requests` |
| Offline | ✅ Works offline | ❌ Requires internet |
| Memory | Limited by local RAM | Server handles it |

**Recommendation:**
- Use **local** for: development, offline work, small files
- Use **API** for: large files (>100MB), simple deployments, avoiding dependency issues

---

## Troubleshooting

### "Model not found"

The CLI looks for the model in:
1. `model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl`
2. `backend/models/XGBoost_1.2.4.pkl`

Solution: Use `--api` flag for remote scoring.

### "Expected stereo WAV file"

The WAV file must have exactly 2 channels:
- Channel 1: EEG signal
- Channel 2: EMG signal

### API timeout

Large files may take time to upload and process.

```bash
# The CLI uses a 5-minute timeout by default
# For very large files, consider splitting or using local scoring
```

### Import errors

Ensure you're running from the project root or have installed the CLI:

```bash
cd /path/to/sleep
pip install -e ./cli
```

---

## Scripting Examples

### Python script integration

```python
import subprocess
import json

# Score and capture JSON output
result = subprocess.run(
    ["mora", "score", "recording.wav", "--json"],
    capture_output=True,
    text=True
)
data = json.loads(result.stdout)
print(f"Wake: {data['summary']['Wake']['percentage']}%")
```

### Bash loop

```bash
#!/bin/bash
for file in recordings/*.wav; do
    echo "Processing $file..."
    mora score "$file" --api -o "output/$(basename "$file" .wav)_scores.csv"
done
```
