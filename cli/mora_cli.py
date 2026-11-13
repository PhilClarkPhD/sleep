#!/usr/bin/env python3
"""
Mora Sleep Scoring CLI

Command-line interface for scoring EEG/EMG recordings.

Usage:
    mora score recording.wav                    # Score locally
    mora score recording.wav --api              # Score via API
    mora score recording.wav -o scores.csv      # Export to CSV

Examples:
    # Score a file locally (requires backend deps + model file)
    mora score path/to/recording.wav

    # Score using the remote API
    mora score path/to/recording.wav --api --api-url https://sleep-production.up.railway.app

    # Score and export to CSV
    mora score path/to/recording.wav -o results.csv

    # Batch score multiple files
    mora batch path/to/folder/ -o output_folder/
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Entry-point bootstrap: ensure the repo root is importable so `sleep_features`
# resolves regardless of how this script is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
for _p in (_REPO_ROOT, _BACKEND_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_API_URL = "https://sleep-production.up.railway.app"

import click

# Version
__version__ = "1.0.0"


@click.group()
@click.version_option(version=__version__)
def cli():
    """Mora Sleep Scoring CLI - Score EEG/EMG recordings for sleep states."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output CSV file path")
@click.option("--api", is_flag=True, help="Use remote API instead of local model")
@click.option("--api-url", default=DEFAULT_API_URL, help="API URL for remote scoring")
@click.option(
    "--api-key",
    envvar="MORA_API_KEY",
    default=None,
    help="API key for the remote server (or set MORA_API_KEY). Only needed if the server requires one.",
)
@click.option(
    "--start-epoch",
    default=2,
    type=int,
    help="Baseline epoch for legacy (baseline-normalized) models; ignored by newer models. Same default as the API (2).",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON instead of table")
def score(
    file: str,
    output: Optional[str],
    api: bool,
    api_url: str,
    api_key: Optional[str],
    start_epoch: int,
    output_json: bool,
):
    """
    Score a WAV file for sleep states.

    FILE: Path to stereo WAV file (EEG on channel 1, EMG on channel 2)
    """
    file_path = Path(file)

    if not file_path.suffix.lower() == ".wav":
        click.echo(f"Error: Expected .wav file, got {file_path.suffix}", err=True)
        sys.exit(1)

    click.echo(f"Scoring: {file_path.name}")

    if api:
        scores, summary = score_via_api(file_path, api_url, start_epoch, api_key)
    else:
        scores, summary = score_locally(file_path, start_epoch)

    # Display results
    if output_json:
        result = {"scores": scores, "summary": summary}
        click.echo(json.dumps(result, indent=2))
    else:
        display_summary(summary)

    # Export if requested
    if output:
        export_to_csv(scores, Path(output))
        click.echo(f"Exported to: {output}")


@cli.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Output folder")
@click.option("--api", is_flag=True, help="Use remote API instead of local model")
@click.option("--api-url", default=DEFAULT_API_URL)
@click.option(
    "--api-key",
    envvar="MORA_API_KEY",
    default=None,
    help="API key for the remote server (or set MORA_API_KEY). Only needed if the server requires one.",
)
@click.option("--start-epoch", default=2, type=int)
def batch(folder: str, output: str, api: bool, api_url: str, api_key: Optional[str], start_epoch: int):
    """
    Score all WAV files in a folder.

    FOLDER: Path to folder containing WAV files
    """
    folder_path = Path(folder)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.WAV"))

    if not wav_files:
        click.echo(f"No WAV files found in {folder}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(wav_files)} WAV files")

    for i, wav_file in enumerate(wav_files, 1):
        click.echo(f"[{i}/{len(wav_files)}] {wav_file.name}...")

        try:
            if api:
                scores, summary = score_via_api(wav_file, api_url, start_epoch, api_key)
            else:
                scores, summary = score_locally(wav_file, start_epoch)
            display_reliability(summary.get("reliability"))

            output_file = output_path / f"{wav_file.stem}_scores.csv"
            export_to_csv(scores, output_file)
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    click.echo(f"Done! Results saved to {output_path}")


@cli.command()
@click.option("--api-url", default=DEFAULT_API_URL)
@click.option(
    "--api-key",
    envvar="MORA_API_KEY",
    default=None,
    help="API key for the remote server (or set MORA_API_KEY). Only needed if the server requires one.",
)
def info(api_url: str, api_key: Optional[str]):
    """Show information about the model served by the API."""
    import requests

    try:
        response = requests.get(
            f"{api_url}/api/v1/model/info", headers=_auth_headers(api_key), timeout=10
        )
        response.raise_for_status()
        data = response.json()

        click.echo("Model Information:")
        click.echo(f"  Name:    {data.get('model_name', 'Unknown')}")
        click.echo(f"  Version: {data.get('model_version', 'Unknown')}")
        click.echo(f"  Classes: {', '.join(data.get('classes', []))}")
        click.echo(f"  Features: {len(data.get('feature_columns', []))} columns")

    except requests.RequestException as e:
        click.echo(f"Error connecting to API: {e}", err=True)
        sys.exit(1)


def score_locally(file_path: Path, start_epoch: int) -> tuple:
    """
    Score a file with the local model, using the SAME code path as the API
    (backend ScoringService): model metadata picks the feature pipeline and
    normalization, signal QC marks bad epochs Unscored, and recording-level
    reliability warnings are computed. Requires `pip install -r requirements.txt`
    from the repo root (backend deps). Model file: MORA_MODEL_PATH, else the
    *.pkl in backend/models/.
    """
    try:
        from app.core.config import settings
        from app.core.model_loader import ModelManager
        from app.services.scoring_service import ScoringService
    except ImportError as e:
        raise click.ClickException(
            f"Local scoring needs the backend dependencies ({e}). "
            "Run `pip install -r requirements.txt` from the repo root, or use --api."
        )

    click.echo(f"  Loading model from {settings.MODEL_PATH}")
    manager = ModelManager()
    manager.load()
    if not manager.is_loaded:
        raise click.ClickException(
            f"Model not found at {settings.MODEL_PATH}. Set MORA_MODEL_PATH or use --api."
        )

    click.echo("  Scoring...")
    result = ScoringService(manager).score(
        file_path.read_bytes(), start_epoch=start_epoch, include_signals=False
    )
    # Default to the UNFILTERED scores, matching the web app's default view.
    scores = [e.score for e in result.unfiltered_epochs]
    summary = calculate_summary(scores)
    summary["reliability"] = result.reliability.model_dump() if result.reliability else None
    summary["model_version"] = manager.version
    return scores, summary


def _auth_headers(api_key: Optional[str]) -> dict:
    return {"X-API-Key": api_key} if api_key else {}


def score_via_api(file_path: Path, api_url: str, start_epoch: int, api_key: Optional[str] = None) -> tuple:
    """Score a file using the remote API."""
    import requests

    click.echo(f"  Uploading to {api_url}...")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "audio/wav")}
        data = {"start_epoch": start_epoch, "include_signals": "false"}

        response = requests.post(
            f"{api_url}/api/v1/score",
            files=files,
            data=data,
            headers=_auth_headers(api_key),
            timeout=300,  # 5 minute timeout for large files
        )

    if response.status_code in (401, 403):
        raise click.ClickException(
            "The API requires a key. Pass --api-key or set MORA_API_KEY."
        )
    if response.status_code != 200:
        raise click.ClickException(f"API error: {response.status_code} - {response.text}")

    result = response.json()
    # Use the UNFILTERED scores, matching the web app's default view.
    scores = [e["score"] for e in result.get("unfiltered_epochs") or result.get("epochs", [])]
    summary = calculate_summary(scores)
    summary["reliability"] = result.get("reliability")
    summary["model_version"] = result.get("model_version")

    return scores, summary


def calculate_summary(scores: list) -> dict:
    """Calculate summary statistics from scores."""
    from collections import Counter

    counts = Counter(scores)
    total = len(scores)
    scorable = total - counts.get("Unscored", 0)

    summary = {}
    for state in ["Wake", "Non REM", "REM", "Unscored"]:
        count = counts.get(state, 0)
        if state == "Unscored" and count == 0:
            continue
        denom = total if state == "Unscored" else scorable
        summary[state] = {
            "count": count,
            "percentage": round(100 * count / denom, 1) if denom > 0 else 0,
        }

    # Add time in each state (assuming 10s epochs)
    for state in summary:
        minutes = summary[state]["count"] * 10 / 60
        summary[state]["minutes"] = round(minutes, 1)

    return summary


def display_summary(summary: dict):
    """Display summary statistics."""
    click.echo("\nSummary:")
    click.echo("-" * 40)
    click.echo(f"{'State':<12} {'Epochs':>8} {'Time':>10} {'%':>8}")
    click.echo("-" * 40)

    for state in ["Wake", "Non REM", "REM", "Unscored"]:
        if state in summary:
            s = summary[state]
            click.echo(
                f"{state:<12} {s['count']:>8} {s['minutes']:>8.1f}m {s['percentage']:>7.1f}%"
            )

    click.echo("-" * 40)
    display_reliability(summary.get("reliability"))


def display_reliability(report: Optional[dict]):
    """Display recording-level reliability warnings (whole-recording, not per-epoch)."""
    if not report:
        return
    warnings = report.get("warnings") or []
    if not warnings:
        ood = report.get("ood_percent")
        ood_txt = f", OOD {ood:.1f}%" if ood is not None else ""
        click.echo(f"Recording quality: OK (QC fail {report.get('qc_fail_percent', 0):.2f}%{ood_txt})")
        return
    click.echo("\nWARNING: this recording may be harder to score. Review results carefully.")
    for w in warnings:
        click.echo(f"  - [{w['code']}] {w['metric']} = {w['value']} (threshold {w['threshold']}): {w['message']}")


def export_to_csv(scores: list, output_path: Path):
    """Export scores to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, score])


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
