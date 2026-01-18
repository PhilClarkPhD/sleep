/**
 * Hypnogram visualization component using Plotly.
 *
 * A hypnogram shows sleep stages over time as a step function.
 * Click anywhere on the plot to navigate to that epoch.
 * Includes light/dark phase background shading when configured.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';
import { SLEEP_COLORS, type SleepState } from '../types/scoring';
import type { Layout } from 'plotly.js';

// Map sleep states to numeric values for y-axis
const SLEEP_Y_VALUES: Record<SleepState, number> = {
  'Wake': 2,
  'REM': 1,
  'Non REM': 0,
  'Unscored': 3,
};

// Phase background colors (very subtle)
const PHASE_COLORS = {
  light: 'rgba(255, 251, 235, 0.7)',  // very light yellow
  dark: 'rgba(229, 231, 235, 0.7)',   // very light gray
};

export function Hypnogram() {
  const {
    epochs,
    summary,
    currentEpoch,
    setCurrentEpoch,
    recordingStartTime,
    lightDarkPhases,
    getTimestampForEpoch,
    getPhaseForEpoch,
  } = useAppStore();

  if (epochs.length === 0) {
    return null;
  }

  const hasTimestamp = !!recordingStartTime;
  const hasPhases = lightDarkPhases.length > 0 && hasTimestamp;

  // Prepare data for Plotly
  // Convert timestamps to hours for readability
  const x = epochs.map(e => e.timestamp_seconds / 3600);
  const y = epochs.map(e => SLEEP_Y_VALUES[e.score as SleepState] ?? 3);

  // Current epoch position (for vertical line indicator)
  const currentTime = epochs[currentEpoch]?.timestamp_seconds / 3600 || 0;

  // Build phase background shapes
  const phaseShapes: Partial<Layout['shapes']>[number][] = [];
  if (hasPhases) {
    let currentPhase = getPhaseForEpoch(0);
    let phaseStartHours = 0;

    for (let i = 1; i <= epochs.length; i++) {
      const phase = i < epochs.length ? getPhaseForEpoch(i) : null;
      if (phase !== currentPhase || i === epochs.length) {
        // Phase changed or end of epochs - close the current phase shape
        if (currentPhase) {
          const phaseEndHours = epochs[i - 1].timestamp_seconds / 3600 + 10 / 3600; // include last epoch
          phaseShapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: phaseStartHours,
            x1: phaseEndHours,
            y0: 0,
            y1: 1,
            fillcolor: PHASE_COLORS[currentPhase],
            line: { width: 0 },
            layer: 'below',
          });
        }
        currentPhase = phase;
        phaseStartHours = i < epochs.length ? epochs[i].timestamp_seconds / 3600 : 0;
      }
    }
  }

  // Build custom tick labels with actual times if available
  const tickCount = Math.min(10, Math.ceil(epochs.length / 360)); // ~1 tick per hour, max 10
  const tickInterval = Math.floor(epochs.length / tickCount);
  const tickvals: number[] = [];
  const ticktext: string[] = [];

  for (let i = 0; i < epochs.length; i += tickInterval) {
    const hours = epochs[i].timestamp_seconds / 3600;
    tickvals.push(hours);

    if (hasTimestamp) {
      const timestamp = getTimestampForEpoch(i);
      if (timestamp) {
        const h = timestamp.getHours().toString().padStart(2, '0');
        const m = timestamp.getMinutes().toString().padStart(2, '0');
        ticktext.push(`${h}:${m}`);
      } else {
        ticktext.push(`${hours.toFixed(1)}h`);
      }
    } else {
      ticktext.push(`${hours.toFixed(1)}h`);
    }
  }

  // Handle click on plot to navigate to epoch
  const handlePlotClick = (event: Plotly.PlotMouseEvent) => {
    if (event.points && event.points.length > 0) {
      const pointIndex = event.points[0].pointIndex;
      if (typeof pointIndex === 'number') {
        setCurrentEpoch(pointIndex);
      }
    }
  };

  return (
    <div className="w-full bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Hypnogram</h2>
        <span className="text-sm text-gray-500">Click to navigate to epoch</span>
      </div>

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
              color: '#1f2937',
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: 'Epoch %{pointIndex}<br>%{text}<br>Time: %{x:.2f} hours<extra></extra>',
            text: epochs.map(e => e.score),
          },
        ]}
        layout={{
          autosize: true,
          height: 200,
          margin: { l: 80, r: 20, t: 20, b: 50 },
          xaxis: {
            title: { text: hasTimestamp ? 'Time' : 'Time (hours)' },
            showgrid: true,
            gridcolor: '#e5e7eb',
            tickmode: 'array' as const,
            tickvals,
            ticktext,
          },
          yaxis: {
            title: { text: 'Sleep State' },
            tickmode: 'array' as const,
            tickvals: [0, 1, 2, 3],
            ticktext: ['NREM', 'REM', 'Wake', '?'],
            range: [-0.5, 3.5],
            showgrid: true,
            gridcolor: '#e5e7eb',
          },
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          hovermode: 'closest',
          // Phase background shapes + vertical line showing current epoch
          shapes: [
            ...phaseShapes,
            {
              type: 'line',
              xref: 'x',
              yref: 'paper',
              x0: currentTime,
              x1: currentTime,
              y0: 0,
              y1: 1,
              line: {
                color: '#ef4444',
                width: 2,
                dash: 'solid',
              },
            },
          ] as Layout['shapes'],
          annotations: [
            {
              x: currentTime,
              y: 1.05,
              xref: 'x',
              yref: 'paper',
              text: `E${currentEpoch}`,
              showarrow: false,
              font: { size: 10, color: '#ef4444' },
            },
          ],
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }}
        style={{ width: '100%' }}
        onClick={handlePlotClick}
      />

      {/* Color legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-4">
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
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded border border-gray-300" style={{ backgroundColor: SLEEP_COLORS['Unscored'] }} />
          <span className="text-sm">Unscored</span>
        </div>
      </div>
    </div>
  );
}
