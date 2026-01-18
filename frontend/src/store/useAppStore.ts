/**
 * Global application state using Zustand.
 *
 * Zustand is a lightweight state management library (similar to Redux but simpler).
 * Think of it like a global Python dictionary that all components can read/write.
 */

import { create } from 'zustand';
import type { EpochScore, ScoringStats, ModelInfo } from '../types/scoring';

export type UploadState = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

interface AppState {
  // Upload state
  uploadState: UploadState;
  uploadProgress: number;
  errorMessage: string | null;

  // File info
  fileName: string | null;

  // Scoring results
  epochs: EpochScore[];
  summary: ScoringStats | null;
  modelVersion: string | null;
  samplerate: number | null;
  baselineEpoch: number;

  // Model info
  modelInfo: ModelInfo | null;

  // Actions (functions to update state)
  setUploadState: (state: UploadState) => void;
  setUploadProgress: (progress: number) => void;
  setError: (message: string | null) => void;
  setFileName: (name: string | null) => void;
  setBaselineEpoch: (epoch: number) => void;
  setScoringResults: (epochs: EpochScore[], summary: ScoringStats, modelVersion: string, samplerate: number) => void;
  setModelInfo: (info: ModelInfo) => void;
  reset: () => void;
}

const initialState = {
  uploadState: 'idle' as UploadState,
  uploadProgress: 0,
  errorMessage: null,
  fileName: null,
  epochs: [],
  summary: null,
  modelVersion: null,
  samplerate: null,
  baselineEpoch: 2,
  modelInfo: null,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setUploadState: (uploadState) => set({ uploadState }),

  setUploadProgress: (uploadProgress) => set({ uploadProgress }),

  setError: (errorMessage) => set({ errorMessage, uploadState: errorMessage ? 'error' : 'idle' }),

  setFileName: (fileName) => set({ fileName }),

  setBaselineEpoch: (baselineEpoch) => set({ baselineEpoch }),

  setScoringResults: (epochs, summary, modelVersion, samplerate) => set({
    epochs,
    summary,
    modelVersion,
    samplerate,
    uploadState: 'complete',
  }),

  setModelInfo: (modelInfo) => set({ modelInfo }),

  reset: () => set(initialState),
}));
