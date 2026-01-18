/**
 * API client for the Mora Sleep Scoring backend.
 *
 * The API URL is configured via VITE_API_URL environment variable:
 * - Development: Uses Vite proxy (/api/v1 -> localhost:8000)
 * - Production: Set VITE_API_URL at build time (e.g., https://api.mora.example.com)
 */

import type { ScoringResponse, ModelInfo, HealthResponse } from '../types/scoring';

// In production, VITE_API_URL should be set to the backend URL
// In development, we use the Vite proxy (empty string = relative URL)
const API_BASE_URL = import.meta.env.VITE_API_URL || '';
const API_BASE = `${API_BASE_URL}/api/v1`;

/**
 * Check if the API is healthy and the model is loaded.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get model information.
 */
export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/model/info`);
  if (!response.ok) {
    throw new Error(`Failed to get model info: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Score a WAV file.
 *
 * @param file - The WAV file to score
 * @param startEpoch - Baseline epoch for normalization (default: 2)
 * @param includeSignals - Whether to include signal data for visualization (default: true)
 * @param onProgress - Optional callback for upload progress (0-100)
 */
export async function scoreFile(
  file: File,
  startEpoch: number = 2,
  includeSignals: boolean = true,
  onProgress?: (percent: number) => void
): Promise<ScoringResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('start_epoch', startEpoch.toString());
  formData.append('include_signals', includeSignals.toString());

  // Use XMLHttpRequest for progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(percent);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch {
          reject(new Error('Failed to parse response'));
        }
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          reject(new Error(error.detail || `Request failed: ${xhr.statusText}`));
        } catch {
          reject(new Error(`Request failed: ${xhr.statusText}`));
        }
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error'));
    });

    xhr.open('POST', `${API_BASE}/score`);
    xhr.send(formData);
  });
}

/**
 * Convert scoring results to CSV format.
 */
export function scoresToCsv(epochs: { epoch: number; score: string; timestamp_seconds: number }[]): string {
  const header = 'epoch,score,timestamp_seconds\n';
  const rows = epochs.map(e => `${e.epoch},${e.score},${e.timestamp_seconds}`).join('\n');
  return header + rows;
}

/**
 * Download a string as a file.
 */
export function downloadFile(content: string, filename: string, mimeType: string = 'text/csv'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
