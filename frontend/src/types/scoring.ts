/**
 * Types matching the backend API schemas.
 */

export interface EpochScore {
  epoch: number;
  score: 'Wake' | 'Non REM' | 'REM';
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
}

export interface ScoringResponse {
  success: boolean;
  epochs: EpochScore[];
  summary: ScoringStats;
  model_version: string;
  samplerate: number;
  baseline_epoch: number;
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
}

export type SleepState = 'Wake' | 'Non REM' | 'REM';

// Color mapping for sleep states
export const SLEEP_COLORS: Record<SleepState, string> = {
  'Wake': '#f59e0b',     // amber-500
  'Non REM': '#3b82f6',  // blue-500
  'REM': '#22c55e',      // green-500
};
