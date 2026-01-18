/**
 * ExportPanel - Import and export scoring data.
 *
 * Features:
 * - Export scores as CSV
 * - Export summary breakdown
 * - Import scores from CSV
 * - Clear all scores
 */

import { useRef } from 'react';
import { useAppStore } from '../store/useAppStore';
import { downloadFile } from '../api/sleepApi';
import type { EpochScore, SleepState } from '../types/scoring';

export function ExportPanel() {
  const {
    epochs,
    summary,
    fileName,
    hasUnsavedChanges,
    importScores,
    clearAllScores,
  } = useAppStore();

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (epochs.length === 0) {
    return null;
  }

  // Export scores as CSV
  const handleExportScores = () => {
    const header = 'epoch,score,timestamp_seconds\n';
    const rows = epochs.map(e => `${e.epoch},${e.score},${e.timestamp_seconds}`).join('\n');
    const csv = header + rows;

    const baseName = fileName?.replace('.wav', '') || 'scores';
    downloadFile(csv, `${baseName}_scores.csv`, 'text/csv');
  };

  // Export summary breakdown
  const handleExportBreakdown = () => {
    if (!summary) return;

    const header = 'Stage,Epochs,Proportion\n';
    const rows = [
      `Wake,${summary.wake_count},${(summary.wake_percent / 100).toFixed(4)}`,
      `Non REM,${summary.nrem_count},${(summary.nrem_percent / 100).toFixed(4)}`,
      `REM,${summary.rem_count},${(summary.rem_percent / 100).toFixed(4)}`,
    ].join('\n');
    const csv = header + rows;

    const baseName = fileName?.replace('.wav', '') || 'breakdown';
    downloadFile(csv, `${baseName}_breakdown.csv`, 'text/csv');
  };

  // Import scores from CSV
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.trim().split('\n');

      // Parse header to find column indices
      const header = lines[0].toLowerCase();
      const columns = header.split(',').map(c => c.trim());
      const epochIdx = columns.findIndex(c => c === 'epoch' || c === 'epoch #');
      const scoreIdx = columns.findIndex(c => c === 'score');

      if (epochIdx === -1 || scoreIdx === -1) {
        alert('CSV must have "epoch" and "score" columns');
        return;
      }

      // Parse data rows
      const importedScores: EpochScore[] = [];
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',').map(c => c.trim());
        let epochNum = parseInt(cols[epochIdx], 10);
        const score = cols[scoreIdx] as SleepState;

        // Handle Sirenia format (1-indexed)
        if (header.includes('epoch #')) {
          epochNum -= 1;
        }

        if (!isNaN(epochNum) && score) {
          // Normalize score names
          let normalizedScore = score;
          if (score.toLowerCase() === 'nrem' || score.toLowerCase() === 'non-rem') {
            normalizedScore = 'Non REM';
          } else if (score.toLowerCase() === 'wake') {
            normalizedScore = 'Wake';
          } else if (score.toLowerCase() === 'rem') {
            normalizedScore = 'REM';
          }

          importedScores.push({
            epoch: epochNum,
            score: normalizedScore as SleepState,
            timestamp_seconds: epochNum * 10,
          });
        }
      }

      if (importedScores.length > 0) {
        importScores(importedScores);
        alert(`Imported ${importedScores.length} scores`);
      } else {
        alert('No valid scores found in file');
      }
    };

    reader.readAsText(file);
    // Reset input so the same file can be imported again
    event.target.value = '';
  };

  // Confirm before clearing
  const handleClearScores = () => {
    if (confirm('Are you sure you want to clear all scores? This cannot be undone.')) {
      clearAllScores();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Import / Export</h2>

      <div className="flex flex-wrap gap-3">
        {/* Export scores */}
        <button
          onClick={handleExportScores}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Export Scores (CSV)
        </button>

        {/* Export breakdown */}
        <button
          onClick={handleExportBreakdown}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Export Breakdown
        </button>

        {/* Import scores */}
        <button
          onClick={handleImportClick}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
        >
          Import Scores
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt"
          onChange={handleFileImport}
          className="hidden"
        />

        {/* Clear scores */}
        <button
          onClick={handleClearScores}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          Clear Scores
        </button>
      </div>

      {/* Unsaved changes warning */}
      {hasUnsavedChanges && (
        <p className="mt-3 text-sm text-amber-600">
          You have unsaved changes. Export your scores to save them.
        </p>
      )}
    </div>
  );
}
