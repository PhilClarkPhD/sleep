/**
 * Global application state using Zustand.
 *
 * Zustand is a lightweight state management library (similar to Redux but simpler).
 * Think of it like a global Python dictionary that all components can read/write.
 */

import { create } from 'zustand';
import type { EpochScore, ScoringStats, ModelInfo, SignalData, SleepState, LightDarkPhase, PhaseType } from '../types/scoring';

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

  // Signal data for visualization
  signalData: SignalData | null;

  // Model info
  modelInfo: ModelInfo | null;

  // Navigation state
  currentEpoch: number;
  windowSize: number;  // Number of epochs visible (1, 3, 5, or 7)

  // Editing state
  editedEpochs: Set<number>;  // Tracks which epochs have been manually edited
  hasUnsavedChanges: boolean;

  // Recording metadata
  recordingStartTime: string | null;  // ISO datetime string
  lightDarkPhases: LightDarkPhase[];

  // Actions (functions to update state)
  setUploadState: (state: UploadState) => void;
  setUploadProgress: (progress: number) => void;
  setError: (message: string | null) => void;
  setFileName: (name: string | null) => void;
  setBaselineEpoch: (epoch: number) => void;
  setScoringResults: (
    epochs: EpochScore[],
    summary: ScoringStats,
    modelVersion: string,
    samplerate: number,
    signalData?: SignalData
  ) => void;
  setModelInfo: (info: ModelInfo) => void;

  // Navigation actions
  setCurrentEpoch: (epoch: number) => void;
  goToNextEpoch: () => void;
  goToPrevEpoch: () => void;
  goToNextREM: () => void;
  setWindowSize: (size: number) => void;

  // Scoring actions
  updateEpochScore: (epoch: number, score: SleepState) => void;
  clearAllScores: () => void;
  importScores: (scores: EpochScore[]) => void;

  // Recording metadata actions
  setRecordingStartTime: (time: string | null) => void;
  addLightDarkPhase: (type: PhaseType, startTime: string, endTime: string) => string | null; // returns error message or null
  updateLightDarkPhase: (id: string, updates: Partial<Omit<LightDarkPhase, 'id'>>) => string | null; // returns error message or null
  removeLightDarkPhase: (id: string) => void;
  clearAllPhases: () => void;
  checkPhaseOverlap: (startTime: string, endTime: string, excludeId?: string) => boolean;
  getPhaseForEpoch: (epochIndex: number) => PhaseType | null;
  getTimestampForEpoch: (epochIndex: number) => Date | null;

  reset: () => void;
}

const initialState = {
  uploadState: 'idle' as UploadState,
  uploadProgress: 0,
  errorMessage: null,
  fileName: null,
  epochs: [] as EpochScore[],
  summary: null,
  modelVersion: null,
  samplerate: null,
  baselineEpoch: 2,
  signalData: null,
  modelInfo: null,
  currentEpoch: 0,
  windowSize: 5,
  editedEpochs: new Set<number>(),
  hasUnsavedChanges: false,
  recordingStartTime: null as string | null,
  lightDarkPhases: [] as LightDarkPhase[],
};

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  setUploadState: (uploadState) => set({ uploadState }),

  setUploadProgress: (uploadProgress) => set({ uploadProgress }),

  setError: (errorMessage) => set({ errorMessage, uploadState: errorMessage ? 'error' : 'idle' }),

  setFileName: (fileName) => set({ fileName }),

  setBaselineEpoch: (baselineEpoch) => set({ baselineEpoch }),

  setScoringResults: (epochs, summary, modelVersion, samplerate, signalData) => set({
    epochs,
    summary,
    modelVersion,
    samplerate,
    signalData: signalData || null,
    uploadState: 'complete',
    currentEpoch: 0,
    editedEpochs: new Set<number>(),
    hasUnsavedChanges: false,
  }),

  setModelInfo: (modelInfo) => set({ modelInfo }),

  // Navigation actions
  setCurrentEpoch: (epoch) => {
    const { epochs } = get();
    if (epoch >= 0 && epoch < epochs.length) {
      set({ currentEpoch: epoch });
    }
  },

  goToNextEpoch: () => {
    const { currentEpoch, epochs } = get();
    if (currentEpoch < epochs.length - 1) {
      set({ currentEpoch: currentEpoch + 1 });
    }
  },

  goToPrevEpoch: () => {
    const { currentEpoch } = get();
    if (currentEpoch > 0) {
      set({ currentEpoch: currentEpoch - 1 });
    }
  },

  goToNextREM: () => {
    const { currentEpoch, epochs } = get();
    // Search forward from current epoch for next REM
    for (let i = currentEpoch + 1; i < epochs.length; i++) {
      if (epochs[i].score === 'REM') {
        set({ currentEpoch: i });
        return;
      }
    }
    // Wrap around to beginning
    for (let i = 0; i < currentEpoch; i++) {
      if (epochs[i].score === 'REM') {
        set({ currentEpoch: i });
        return;
      }
    }
  },

  setWindowSize: (windowSize) => set({ windowSize }),

  // Scoring actions
  updateEpochScore: (epoch, score) => {
    const { epochs, editedEpochs, summary } = get();
    if (epoch < 0 || epoch >= epochs.length) return;

    const oldScore = epochs[epoch].score;
    if (oldScore === score) return;  // No change

    // Update the epoch
    const newEpochs = [...epochs];
    newEpochs[epoch] = { ...newEpochs[epoch], score };

    // Update the edited epochs set
    const newEdited = new Set(editedEpochs);
    newEdited.add(epoch);

    // Recalculate summary
    let newSummary = summary;
    if (summary) {
      const wakeDelta = (score === 'Wake' ? 1 : 0) - (oldScore === 'Wake' ? 1 : 0);
      const nremDelta = (score === 'Non REM' ? 1 : 0) - (oldScore === 'Non REM' ? 1 : 0);
      const remDelta = (score === 'REM' ? 1 : 0) - (oldScore === 'REM' ? 1 : 0);

      const newWakeCount = summary.wake_count + wakeDelta;
      const newNremCount = summary.nrem_count + nremDelta;
      const newRemCount = summary.rem_count + remDelta;
      const total = summary.total_epochs;

      newSummary = {
        ...summary,
        wake_count: newWakeCount,
        wake_percent: Math.round((newWakeCount / total) * 10000) / 100,
        nrem_count: newNremCount,
        nrem_percent: Math.round((newNremCount / total) * 10000) / 100,
        rem_count: newRemCount,
        rem_percent: Math.round((newRemCount / total) * 10000) / 100,
      };
    }

    set({
      epochs: newEpochs,
      editedEpochs: newEdited,
      summary: newSummary,
      hasUnsavedChanges: true,
    });
  },

  clearAllScores: () => {
    const { epochs } = get();
    const clearedEpochs = epochs.map(e => ({ ...e, score: 'Unscored' as SleepState }));
    set({
      epochs: clearedEpochs,
      summary: {
        total_epochs: epochs.length,
        wake_count: 0,
        wake_percent: 0,
        nrem_count: 0,
        nrem_percent: 0,
        rem_count: 0,
        rem_percent: 0,
        recording_duration_seconds: epochs.length * 10,
        recording_duration_hours: (epochs.length * 10) / 3600,
      },
      editedEpochs: new Set<number>(),
      hasUnsavedChanges: true,
    });
  },

  importScores: (scores) => {
    const { epochs } = get();
    if (scores.length === 0) return;

    // Map imported scores to existing epochs
    const newEpochs = epochs.map((e, i) => {
      const imported = scores.find(s => s.epoch === i);
      return imported ? { ...e, score: imported.score } : e;
    });

    // Recalculate summary
    const wakeCount = newEpochs.filter(e => e.score === 'Wake').length;
    const nremCount = newEpochs.filter(e => e.score === 'Non REM').length;
    const remCount = newEpochs.filter(e => e.score === 'REM').length;
    const total = newEpochs.length;

    set({
      epochs: newEpochs,
      summary: {
        total_epochs: total,
        wake_count: wakeCount,
        wake_percent: Math.round((wakeCount / total) * 10000) / 100,
        nrem_count: nremCount,
        nrem_percent: Math.round((nremCount / total) * 10000) / 100,
        rem_count: remCount,
        rem_percent: Math.round((remCount / total) * 10000) / 100,
        recording_duration_seconds: total * 10,
        recording_duration_hours: (total * 10) / 3600,
      },
      hasUnsavedChanges: true,
    });
  },

  // Recording metadata actions
  setRecordingStartTime: (recordingStartTime) => set({ recordingStartTime }),

  // Helper to convert time string to minutes since midnight
  checkPhaseOverlap: (startTime, endTime, excludeId) => {
    const { lightDarkPhases } = get();

    const timeToMinutes = (time: string) => {
      const [h, m] = time.split(':').map(Number);
      return h * 60 + m;
    };

    const newStart = timeToMinutes(startTime);
    const newEnd = timeToMinutes(endTime);

    // Check overlap with each existing phase
    for (const phase of lightDarkPhases) {
      if (excludeId && phase.id === excludeId) continue;

      const existingStart = timeToMinutes(phase.startTime);
      const existingEnd = timeToMinutes(phase.endTime);

      // Handle phases that span midnight
      const newSpansMidnight = newStart > newEnd;
      const existingSpansMidnight = existingStart > existingEnd;

      // Convert to ranges for easier comparison
      // For midnight-spanning phases, we check two separate ranges
      const newRanges: [number, number][] = newSpansMidnight
        ? [[newStart, 1440], [0, newEnd]]
        : [[newStart, newEnd]];
      const existingRanges: [number, number][] = existingSpansMidnight
        ? [[existingStart, 1440], [0, existingEnd]]
        : [[existingStart, existingEnd]];

      // Check if any ranges overlap
      for (const [ns, ne] of newRanges) {
        for (const [es, ee] of existingRanges) {
          // Two ranges overlap if start1 < end2 AND start2 < end1
          if (ns < ee && es < ne) {
            return true; // Overlap detected
          }
        }
      }
    }
    return false;
  },

  addLightDarkPhase: (type, startTime, endTime) => {
    // Check for overlap
    if (get().checkPhaseOverlap(startTime, endTime)) {
      return 'This phase overlaps with an existing phase';
    }

    const { lightDarkPhases } = get();
    const newPhase: LightDarkPhase = {
      id: `phase-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      startTime,
      endTime,
    };
    set({ lightDarkPhases: [...lightDarkPhases, newPhase] });
    return null;
  },

  updateLightDarkPhase: (id, updates) => {
    const { lightDarkPhases } = get();
    const existingPhase = lightDarkPhases.find(p => p.id === id);
    if (!existingPhase) return 'Phase not found';

    // Build the updated phase to check for overlaps
    const updatedStartTime = updates.startTime ?? existingPhase.startTime;
    const updatedEndTime = updates.endTime ?? existingPhase.endTime;

    // Check for overlap (excluding this phase)
    if (get().checkPhaseOverlap(updatedStartTime, updatedEndTime, id)) {
      return 'This change would create an overlap with another phase';
    }

    set({
      lightDarkPhases: lightDarkPhases.map(phase =>
        phase.id === id ? { ...phase, ...updates } : phase
      ),
    });
    return null;
  },

  removeLightDarkPhase: (id) => {
    const { lightDarkPhases } = get();
    set({ lightDarkPhases: lightDarkPhases.filter(phase => phase.id !== id) });
  },

  clearAllPhases: () => {
    set({ lightDarkPhases: [] });
  },

  getTimestampForEpoch: (epochIndex) => {
    const { recordingStartTime, epochs } = get();
    if (!recordingStartTime || epochIndex < 0 || epochIndex >= epochs.length) {
      return null;
    }
    const startDate = new Date(recordingStartTime);
    const epochOffsetMs = epochs[epochIndex].timestamp_seconds * 1000;
    return new Date(startDate.getTime() + epochOffsetMs);
  },

  getPhaseForEpoch: (epochIndex) => {
    const { lightDarkPhases } = get();
    const epochTimestamp = get().getTimestampForEpoch(epochIndex);
    if (!epochTimestamp || lightDarkPhases.length === 0) {
      return null;
    }

    const epochHour = epochTimestamp.getHours();
    const epochMinute = epochTimestamp.getMinutes();
    const epochTimeMinutes = epochHour * 60 + epochMinute;

    for (const phase of lightDarkPhases) {
      const [startH, startM] = phase.startTime.split(':').map(Number);
      const [endH, endM] = phase.endTime.split(':').map(Number);
      const startMinutes = startH * 60 + startM;
      const endMinutes = endH * 60 + endM;

      // Handle phases that span midnight
      if (startMinutes <= endMinutes) {
        // Normal case: phase within same day
        if (epochTimeMinutes >= startMinutes && epochTimeMinutes < endMinutes) {
          return phase.type;
        }
      } else {
        // Phase spans midnight (e.g., 22:00 - 06:00)
        if (epochTimeMinutes >= startMinutes || epochTimeMinutes < endMinutes) {
          return phase.type;
        }
      }
    }
    return null;
  },

  reset: () => set({
    ...initialState,
    editedEpochs: new Set<number>(),
    lightDarkPhases: [],
  }),
}));
