/**
 * Types matching the backend API schemas.
 */

// Sleep state type - moved to top as it's used throughout
export type SleepState = 'Wake' | 'Non REM' | 'REM' | 'Unscored';

export interface EpochScore {
  epoch: number;
  score: SleepState;
  timestamp_seconds: number;
}

export interface ScoringStats {
  total_epochs: number;
  wake_count: number;
  wake_percent: number;
  nrem_count: number;
  nrem_percent: number;
  rem_count: number;
  rem_percent: number;
  recording_duration_seconds: number;
  recording_duration_hours: number;
  unscorable_count?: number;    // Epochs marked Unscored by signal QC
  unscorable_percent?: number;
}

/**
 * Signal data for visualization (from backend).
 */
export interface SignalData {
  eeg: number[][];           // EEG signal per epoch (downsampled)
  emg: number[][];           // EMG signal per epoch (downsampled)
  power_spectrum: number[][]; // Power spectrum per epoch (0-50Hz)
  time_axis: number[];       // Time points within epoch (seconds)
  freq_axis: number[];       // Frequency axis for power spectrum (Hz)
  delta_power: number[];     // Relative delta power per epoch
  theta_power: number[];     // Relative theta power per epoch
}

/**
 * Recording-level reliability warnings from the backend. These describe the
 * whole recording (noisy, weak EMG contrast, unlike training data), not
 * individual epochs.
 */
export interface ReliabilityWarning {
  code: 'noisy' | 'weak_emg_contrast' | 'out_of_distribution' | string;
  metric: string;
  value: number;
  threshold: number;
  message: string;
}

export interface RecordingReliability {
  level: 'ok' | 'warning';
  qc_fail_percent: number;
  emg_range_p05_p50: number | null;
  ood_percent: number | null;
  warnings: ReliabilityWarning[];
}

export interface ScoringResponse {
  success: boolean;
  epochs: EpochScore[];
  unfiltered_epochs: EpochScore[];
  summary: ScoringStats;
  model_version: string;
  samplerate: number;
  baseline_epoch: number;
  reliability?: RecordingReliability | null;  // Recording-level warnings
  signal_data?: SignalData;  // Optional signal data for visualization
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  feature_columns: string[];
  classes: string[];
  notes: string | null;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  model_loaded: boolean;
  version: string;
  auth_required: boolean;
}

// Light/Dark phase definition
export type PhaseType = 'light' | 'dark';

export interface LightDarkPhase {
  id: string;           // Unique identifier for React keys
  type: PhaseType;
  startTime: string;    // HH:MM format (24-hour)
  endTime: string;      // HH:MM format (24-hour)
}

// Color mapping for sleep states
export const SLEEP_COLORS: Record<SleepState, string> = {
  'Wake': '#f59e0b',       // amber-500
  'Non REM': '#3b82f6',    // blue-500
  'REM': '#22c55e',        // green-500
  'Unscored': '#9ca3af',   // gray-400
};

// Lighter colors for epoch shading (with transparency)
export const SLEEP_COLORS_LIGHT: Record<SleepState, string> = {
  'Wake': 'rgba(245, 158, 11, 0.2)',      // amber with alpha
  'Non REM': 'rgba(59, 130, 246, 0.2)',   // blue with alpha
  'REM': 'rgba(34, 197, 94, 0.2)',        // green with alpha
  'Unscored': 'rgba(156, 163, 175, 0.2)', // gray with alpha
};

// Darker colors for current epoch highlighting
export const SLEEP_COLORS_DARK: Record<SleepState, string> = {
  'Wake': 'rgba(245, 158, 11, 0.5)',      // amber darker
  'Non REM': 'rgba(59, 130, 246, 0.5)',   // blue darker
  'REM': 'rgba(34, 197, 94, 0.5)',        // green darker
  'Unscored': 'rgba(156, 163, 175, 0.5)', // gray darker
};
