/**
 * EpochViewer - Main component for viewing and scoring epochs.
 *
 * Contains:
 * - Navigation controls (prev/next, find epoch, next REM)
 * - Window size selector
 * - EEG and EMG time series plots
 * - Power spectrum and relative power charts
 * - Manual scoring buttons
 */

import { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { SignalPlot } from './SignalPlot';
import { PowerSpectrum } from './PowerSpectrum';
import { RelativePower } from './RelativePower';
import { SLEEP_COLORS } from '../types/scoring';
import type { SleepState } from '../types/scoring';

export function EpochViewer() {
  const {
    epochs,
    currentEpoch,
    windowSize,
    signalData,
    setCurrentEpoch,
    goToNextEpoch,
    goToPrevEpoch,
    goToNextREM,
    setWindowSize,
    updateEpochScore,
    hasUnsavedChanges,
  } = useAppStore();

  const [findEpochInput, setFindEpochInput] = useState('');
  const [showFindDialog, setShowFindDialog] = useState(false);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't capture if user is typing in an input
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault();
        goToPrevEpoch();
        break;
      case 'ArrowRight':
        e.preventDefault();
        goToNextEpoch();
        break;
      case 'w':
      case 'W':
        e.preventDefault();
        updateEpochScore(currentEpoch, 'Wake');
        goToNextEpoch();
        break;
      case 'e':
      case 'E':
        e.preventDefault();
        updateEpochScore(currentEpoch, 'Non REM');
        goToNextEpoch();
        break;
      case 'r':
      case 'R':
        e.preventDefault();
        updateEpochScore(currentEpoch, 'REM');
        goToNextEpoch();
        break;
      case 't':
      case 'T':
        e.preventDefault();
        updateEpochScore(currentEpoch, 'Unscored');
        goToNextEpoch();
        break;
      case 'f':
      case 'F':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          setShowFindDialog(true);
        }
        break;
    }
  }, [currentEpoch, goToPrevEpoch, goToNextEpoch, updateEpochScore]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleFindEpoch = () => {
    const epoch = parseInt(findEpochInput, 10);
    if (!isNaN(epoch) && epoch >= 0 && epoch < epochs.length) {
      setCurrentEpoch(epoch);
      setShowFindDialog(false);
      setFindEpochInput('');
    }
  };

  if (epochs.length === 0 || !signalData) {
    return null;
  }

  const currentScore = epochs[currentEpoch]?.score as SleepState || 'Unscored';

  return (
    <div className="bg-white rounded-lg shadow-md p-4 space-y-4">
      {/* Header with navigation and epoch info */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Navigation controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={goToPrevEpoch}
            disabled={currentEpoch === 0}
            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed rounded"
            title="Previous epoch (←)"
          >
            ← Prev
          </button>

          <div className="px-3 py-1 bg-gray-50 rounded font-mono">
            Epoch {currentEpoch} / {epochs.length - 1}
          </div>

          <button
            onClick={goToNextEpoch}
            disabled={currentEpoch >= epochs.length - 1}
            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed rounded"
            title="Next epoch (→)"
          >
            Next →
          </button>

          <button
            onClick={() => setShowFindDialog(true)}
            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded"
            title="Find epoch (Ctrl+F)"
          >
            Find
          </button>

          <button
            onClick={goToNextREM}
            className="px-3 py-1 bg-green-100 hover:bg-green-200 text-green-800 rounded"
            title="Jump to next REM epoch"
          >
            Next REM
          </button>
        </div>

        {/* Window size selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="windowSize" className="text-sm text-gray-600">Window:</label>
          <select
            id="windowSize"
            value={windowSize}
            onChange={(e) => setWindowSize(parseInt(e.target.value, 10))}
            className="px-2 py-1 border rounded"
          >
            <option value={1}>1 epoch</option>
            <option value={3}>3 epochs</option>
            <option value={5}>5 epochs</option>
            <option value={7}>7 epochs</option>
          </select>
        </div>

        {/* Current score display */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Score:</span>
          <span
            className="px-3 py-1 rounded font-medium"
            style={{
              backgroundColor: SLEEP_COLORS[currentScore] + '30',
              color: SLEEP_COLORS[currentScore],
            }}
          >
            {currentScore}
          </span>
          {hasUnsavedChanges && (
            <span className="text-xs text-amber-600">(unsaved)</span>
          )}
        </div>
      </div>

      {/* Signal plots */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* EEG plot - takes 3 columns */}
        <div className="lg:col-span-3">
          <SignalPlot title="EEG" signalType="eeg" height={180} />
        </div>

        {/* Power spectrum - takes 1 column */}
        <div className="lg:col-span-1">
          <PowerSpectrum height={180} />
        </div>

        {/* EMG plot - takes 3 columns */}
        <div className="lg:col-span-3">
          <SignalPlot title="EMG" signalType="emg" height={180} />
        </div>

        {/* Relative power - takes 1 column */}
        <div className="lg:col-span-1">
          <RelativePower height={180} />
        </div>
      </div>

      {/* Manual scoring buttons */}
      <div className="flex flex-wrap items-center gap-4 pt-2 border-t">
        <span className="text-sm text-gray-600">Score as:</span>
        {(['Wake', 'Non REM', 'REM', 'Unscored'] as SleepState[]).map((state) => (
          <button
            key={state}
            onClick={() => {
              updateEpochScore(currentEpoch, state);
              goToNextEpoch();
            }}
            className="px-4 py-2 rounded font-medium transition-colors"
            style={{
              backgroundColor: currentScore === state ? SLEEP_COLORS[state] : SLEEP_COLORS[state] + '20',
              color: currentScore === state ? 'white' : SLEEP_COLORS[state],
            }}
          >
            {state} ({state === 'Wake' ? 'W' : state === 'Non REM' ? 'E' : state === 'REM' ? 'R' : 'T'})
          </button>
        ))}
      </div>

      {/* Keyboard shortcuts help */}
      <div className="text-xs text-gray-500 pt-2">
        Keyboard shortcuts: ← → (navigate) | W (Wake) | E (Non REM) | R (REM) | T (Unscored) | Ctrl+F (find)
      </div>

      {/* Find epoch dialog */}
      {showFindDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-4 shadow-xl">
            <h3 className="font-medium mb-2">Find Epoch</h3>
            <div className="flex gap-2">
              <input
                type="number"
                min={0}
                max={epochs.length - 1}
                value={findEpochInput}
                onChange={(e) => setFindEpochInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleFindEpoch()}
                placeholder={`0 - ${epochs.length - 1}`}
                className="px-3 py-1 border rounded w-32"
                autoFocus
              />
              <button
                onClick={handleFindEpoch}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Go
              </button>
              <button
                onClick={() => setShowFindDialog(false)}
                className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
