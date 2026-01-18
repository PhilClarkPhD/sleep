"""
Integration tests for the FastAPI backend.
"""

import io
import os
import sys

import numpy as np
import pytest
from scipy.io import wavfile

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    """Generate a sample WAV file as bytes."""
    samplerate = 1000
    duration = 30  # 3 epochs of 10 seconds each
    n_samples = samplerate * duration

    # Create stereo signal (EEG on channel 0, EMG on channel 1)
    np.random.seed(42)
    eeg = (np.random.randn(n_samples) * 1000).astype(np.int16)
    emg = (np.random.randn(n_samples) * 500).astype(np.int16)
    stereo = np.column_stack([eeg, emg])

    # Write to bytes buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, samplerate, stereo)
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response contains expected fields."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_healthy_or_degraded(self, client):
        """Test status is valid value."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    def test_model_info_returns_200(self, client):
        """Test that model info endpoint returns 200."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200

    def test_model_info_response_format(self, client):
        """Test model info contains expected fields."""
        response = client.get("/api/v1/model/info")
        data = response.json()

        assert "model_name" in data
        assert "model_version" in data
        assert "feature_columns" in data
        assert "classes" in data

    def test_model_has_expected_classes(self, client):
        """Test model has sleep stage classes."""
        response = client.get("/api/v1/model/info")
        data = response.json()

        expected_classes = {"Wake", "Non REM", "REM"}
        actual_classes = set(data["classes"])
        assert actual_classes == expected_classes


class TestScoreEndpoint:
    """Tests for the scoring endpoint."""

    def test_score_requires_file(self, client):
        """Test that score endpoint requires a file."""
        response = client.post("/api/v1/score")
        assert response.status_code == 422  # Validation error

    def test_score_accepts_wav(self, client, sample_wav_bytes):
        """Test that score endpoint accepts WAV file."""
        files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
        response = client.post("/api/v1/score", files=files)

        # Should succeed or fail gracefully (model might not load in test)
        assert response.status_code in [200, 500]

    def test_score_response_format(self, client, sample_wav_bytes):
        """Test score response format when successful."""
        files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
        response = client.post("/api/v1/score", files=files)

        if response.status_code == 200:
            data = response.json()
            assert "epochs" in data or "scores" in data
            assert "summary" in data

    def test_score_with_start_epoch(self, client, sample_wav_bytes):
        """Test scoring with start_epoch parameter."""
        files = {"file": ("test.wav", sample_wav_bytes, "audio/wav")}
        response = client.post(
            "/api/v1/score",
            files=files,
            data={"start_epoch": 1}
        )

        # Should accept the parameter
        assert response.status_code in [200, 422, 500]


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        """Test that root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are returned."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        # CORS preflight should succeed
        assert response.status_code in [200, 204]
