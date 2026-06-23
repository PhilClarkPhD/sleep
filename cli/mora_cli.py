#!/usr/bin/env python3
"""
Mora Sleep Scoring CLI

Command-line interface for scoring EEG/EMG recordings.

Usage:
    mora score recording.wav                    # Score locally
    mora score recording.wav --api              # Score via API
    mora score recording.wav -o scores.csv      # Export to CSV

Examples:
    # Score a file locally (requires model file)
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
@click.option(
    "--api-url",
    default="https://sleep-production.up.railway.app",
    help="API URL for remote scoring",
)
@click.option(
    "--start-epoch",
    default=9,
    type=int,
    help="Epoch to use for baseline normalization (default: 9)",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON instead of table")
def score(
    file: str,
    output: Optional[str],
    api: bool,
    api_url: str,
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
        scores, summary = score_via_api(file_path, api_url, start_epoch)
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
@click.option("--api-url", default="https://sleep-production.up.railway.app")
@click.option("--start-epoch", default=9, type=int)
def batch(folder: str, output: str, api: bool, api_url: str, start_epoch: int):
    """
    Score all WAV files in a folder.

    FOLDER: Path to folder containing WAV files
    """
    import os

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
                scores, _ = score_via_api(wav_file, api_url, start_epoch)
            else:
                scores, _ = score_locally(wav_file, start_epoch)

            output_file = output_path / f"{wav_file.stem}_scores.csv"
            export_to_csv(scores, output_file)
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    click.echo(f"Done! Results saved to {output_path}")


@cli.command()
@click.option("--api-url", default="https://sleep-production.up.railway.app")
def info(api_url: str):
    """Show information about the model."""
    import requests

    try:
        response = requests.get(f"{api_url}/api/v1/model/info", timeout=10)
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
    """Score a file using the local model."""
    import pandas as pd
    from scipy.io import wavfile

    # Import local modules
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app.ml.sleep_functions import apply_rule_based_filter, generate_features

    # Try to find and load model
    model, label_encoder = load_local_model()

    # Load WAV file
    click.echo("  Loading WAV file...")
    samplerate, data = wavfile.read(file_path)

    if len(data.shape) != 2 or data.shape[1] != 2:
        raise ValueError("Expected stereo WAV file (2 channels)")

    df = pd.DataFrame({"eeg": data[:, 0], "emg": data[:, 1]})

    # Extract features
    click.echo("  Extracting features...")
    features = generate_features(df, start_epoch=start_epoch)

    # Get feature columns from model config
    feature_cols = [
        "EEG_quantile_80", "EEG_ptp", "EEG_ss", "EMG_std",
        "EMG_events", "EMG_ptp", "delta_rel", "theta_rel", "theta_over_delta"
    ]
    X = features[feature_cols]

    # Predict
    click.echo("  Scoring...")
    predictions = model.predict(X)
    decoded = label_encoder.inverse_transform(predictions)

    # Apply rule-based filter
    filtered = apply_rule_based_filter(decoded)

    # Calculate summary
    summary = calculate_summary(filtered)

    return filtered, summary


def score_via_api(file_path: Path, api_url: str, start_epoch: int) -> tuple:
    """Score a file using the remote API."""
    import requests

    click.echo(f"  Uploading to {api_url}...")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "audio/wav")}
        data = {"start_epoch": start_epoch}

        response = requests.post(
            f"{api_url}/api/v1/score",
            files=files,
            data=data,
            timeout=300,  # 5 minute timeout for large files
        )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result = response.json()
    scores = result.get("scores", [])
    summary = result.get("summary", {})

    return scores, summary


def load_local_model():
    """Load the local XGBoost model."""
    import joblib

    # Try multiple paths
    possible_paths = [
        Path(__file__).parent.parent / "model_artifacts" / "XGBoost_1.2.4" / "XGBoost_1.2.4.pkl",
        Path(__file__).parent.parent / "backend" / "models" / "XGBoost_1.2.4.pkl",
    ]

    for model_path in possible_paths:
        if model_path.exists():
            click.echo(f"  Loading model from {model_path}")
            model, metadata, label_encoder = joblib.load(model_path)
            return model, label_encoder

    raise FileNotFoundError(
        "Model not found. Try using --api flag for remote scoring, or ensure model file exists."
    )


def calculate_summary(scores: list) -> dict:
    """Calculate summary statistics from scores."""
    from collections import Counter

    counts = Counter(scores)
    total = len(scores)

    summary = {}
    for state in ["Wake", "Non REM", "REM"]:
        count = counts.get(state, 0)
        summary[state] = {
            "count": count,
            "percentage": round(100 * count / total, 1) if total > 0 else 0,
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

    for state in ["Wake", "Non REM", "REM"]:
        if state in summary:
            s = summary[state]
            click.echo(
                f"{state:<12} {s['count']:>8} {s['minutes']:>8.1f}m {s['percentage']:>7.1f}%"
            )

    click.echo("-" * 40)


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
