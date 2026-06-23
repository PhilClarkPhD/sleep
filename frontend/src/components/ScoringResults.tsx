/**
 * Summary statistics display component.
 */

import { useAppStore } from '../store/useAppStore';
import { SLEEP_COLORS } from '../types/scoring';

export function ScoringResults() {
  const { summary, modelVersion, samplerate, fileName, showFiltered, unfilteredEpochs, toggleFilter } = useAppStore();
  const hasUnfilteredData = unfilteredEpochs.length > 0;

  if (!summary) {
    return null;
  }

  const stats = [
    {
      label: 'Total Epochs',
      value: summary.total_epochs.toLocaleString(),
      subtext: `(${summary.recording_duration_hours.toFixed(2)} hours)`,
    },
    {
      label: 'Wake',
      value: summary.wake_count.toLocaleString(),
      percent: summary.wake_percent,
      color: SLEEP_COLORS['Wake'],
    },
    {
      label: 'NREM',
      value: summary.nrem_count.toLocaleString(),
      percent: summary.nrem_percent,
      color: SLEEP_COLORS['Non REM'],
    },
    {
      label: 'REM',
      value: summary.rem_count.toLocaleString(),
      percent: summary.rem_percent,
      color: SLEEP_COLORS['REM'],
    },
  ];

  return (
    <div className="w-full bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-lg font-semibold">Scoring Results</h2>
        <div className="text-right text-sm text-gray-500">
          <div>Model: XGBoost v{modelVersion}</div>
          <div>Sample rate: {samplerate} Hz</div>
          {hasUnfilteredData && (
            <label className="flex items-center gap-2 mt-2 cursor-pointer justify-end">
              <span className={showFiltered ? 'font-medium text-gray-700' : 'text-gray-400'}>
                Post-processing filter
              </span>
              <button
                onClick={toggleFilter}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                  showFiltered ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                    showFiltered ? 'translate-x-4.5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </label>
          )}
        </div>
      </div>

      {fileName && (
        <p className="text-sm text-gray-600 mb-4">
          File: <span className="font-medium">{fileName}</span>
        </p>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="p-4 rounded-lg bg-gray-50 border border-gray-100"
          >
            <div className="text-sm text-gray-500 flex items-center gap-2">
              {stat.color && (
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: stat.color }}
                />
              )}
              {stat.label}
            </div>
            <div className="text-2xl font-semibold mt-1">{stat.value}</div>
            {stat.percent !== undefined && (
              <div className="text-sm text-gray-500">{stat.percent.toFixed(1)}%</div>
            )}
            {stat.subtext && (
              <div className="text-sm text-gray-500">{stat.subtext}</div>
            )}
          </div>
        ))}
      </div>

      {/* Pie chart representation */}
      <div className="mt-6">
        <div className="h-4 rounded-full overflow-hidden flex">
          <div
            style={{
              width: `${summary.wake_percent}%`,
              backgroundColor: SLEEP_COLORS['Wake'],
            }}
            title={`Wake: ${summary.wake_percent.toFixed(1)}%`}
          />
          <div
            style={{
              width: `${summary.nrem_percent}%`,
              backgroundColor: SLEEP_COLORS['Non REM'],
            }}
            title={`NREM: ${summary.nrem_percent.toFixed(1)}%`}
          />
          <div
            style={{
              width: `${summary.rem_percent}%`,
              backgroundColor: SLEEP_COLORS['REM'],
            }}
            title={`REM: ${summary.rem_percent.toFixed(1)}%`}
          />
        </div>
      </div>
    </div>
  );
}
