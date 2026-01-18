/**
 * Hypnogram visualization component using Plotly.
 *
 * A hypnogram shows sleep stages over time as a step function.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';
import { SLEEP_COLORS, type SleepState } from '../types/scoring';

// Map sleep states to numeric values for y-axis
const SLEEP_Y_VALUES: Record<SleepState, number> = {
  'Wake': 2,
  'REM': 1,
  'Non REM': 0,
};

export function Hypnogram() {
  const { epochs, summary } = useAppStore();

  if (epochs.length === 0) {
    return null;
  }

  // Prepare data for Plotly
  // Convert timestamps to hours for readability
  const x = epochs.map(e => e.timestamp_seconds / 3600);
  const y = epochs.map(e => SLEEP_Y_VALUES[e.score as SleepState]);

  return (
    <div className="w-full bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Hypnogram</h2>

      <Plot
        data={[
          {
            x,
            y,
            type: 'scatter',
            mode: 'lines',
            line: {
              shape: 'hv', // Step function (horizontal then vertical)
              width: 2,
              color: '#3b82f6',
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: '%{text}<br>Time: %{x:.2f} hours<extra></extra>',
            text: epochs.map(e => e.score),
          },
        ]}
        layout={{
          autosize: true,
          height: 200,
          margin: { l: 80, r: 20, t: 20, b: 50 },
          xaxis: {
            title: { text: 'Time (hours)' },
            showgrid: true,
            gridcolor: '#e5e7eb',
          },
          yaxis: {
            title: { text: 'Sleep State' },
            tickmode: 'array',
            tickvals: [0, 1, 2],
            ticktext: ['NREM', 'REM', 'Wake'],
            range: [-0.5, 2.5],
            showgrid: true,
            gridcolor: '#e5e7eb',
          },
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          hovermode: 'x unified',
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }}
        style={{ width: '100%' }}
      />

      {/* Color legend */}
      <div className="flex justify-center gap-6 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: SLEEP_COLORS['Wake'] }} />
          <span className="text-sm">Wake ({summary?.wake_percent.toFixed(1)}%)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: SLEEP_COLORS['Non REM'] }} />
          <span className="text-sm">NREM ({summary?.nrem_percent.toFixed(1)}%)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: SLEEP_COLORS['REM'] }} />
          <span className="text-sm">REM ({summary?.rem_percent.toFixed(1)}%)</span>
        </div>
      </div>
    </div>
  );
}
