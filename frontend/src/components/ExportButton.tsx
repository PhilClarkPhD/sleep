/**
 * Export button component for downloading results as CSV.
 */

import { useAppStore } from '../store/useAppStore';
import { scoresToCsv, downloadFile } from '../api/sleepApi';

export function ExportButton() {
  const { epochs, fileName, uploadState } = useAppStore();

  if (uploadState !== 'complete' || epochs.length === 0) {
    return null;
  }

  const handleExport = () => {
    const csv = scoresToCsv(epochs);
    const exportFileName = fileName
      ? fileName.replace(/\.wav$/i, '_scores.csv')
      : 'sleep_scores.csv';
    downloadFile(csv, exportFileName);
  };

  return (
    <button
      onClick={handleExport}
      className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Export CSV
    </button>
  );
}
