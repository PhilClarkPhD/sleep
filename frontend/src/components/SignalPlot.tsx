/**
 * SignalPlot - Reusable time series plot for EEG/EMG signals.
 *
 * Displays signal data with epoch shading based on sleep state.
 * Includes light/dark phase background shading when configured.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';
import { SLEEP_COLORS_LIGHT, SLEEP_COLORS_DARK } from '../types/scoring';
import type { SleepState } from '../types/scoring';
import type { Layout } from 'plotly.js';

// Phase background colors (very subtle)
const PHASE_COLORS = {
  light: 'rgba(255, 251, 235, 0.6)',  // very light yellow
  dark: 'rgba(229, 231, 235, 0.6)',   // very light gray
};

interface SignalPlotProps {
  title: string;
  signalType: 'eeg' | 'emg';
  height?: number;
}

export function SignalPlot({ title, signalType, height = 200 }: SignalPlotProps) {
  const {
    signalData,
    epochs,
    currentEpoch,
    windowSize,
    recordingStartTime,
    lightDarkPhases,
    getTimestampForEpoch,
    getPhaseForEpoch,
  } = useAppStore();

  if (!signalData || epochs.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 flex items-center justify-center" style={{ height }}>
        <p className="text-gray-500">No signal data available</p>
      </div>
    );
  }

  // Get the epochs to display based on window size
  const halfWindow = Math.floor(windowSize / 2);
  const startEpoch = Math.max(0, currentEpoch - halfWindow);
  const endEpoch = Math.min(epochs.length - 1, currentEpoch + halfWindow);

  // Build the data for the plot
  const signal = signalType === 'eeg' ? signalData.eeg : signalData.emg;
  const timeAxis = signalData.time_axis;
  const epochDuration = 10; // seconds
  const hasPhases = lightDarkPhases.length > 0 && !!recordingStartTime;

  // Concatenate signals for visible epochs
  const xData: number[] = [];
  const yData: number[] = [];
  const shapes: Partial<Layout['shapes']>[number][] = [];
  const tickvals: number[] = [];
  const ticktext: string[] = [];

  // First pass: add phase background shapes (behind everything)
  if (hasPhases) {
    for (let i = startEpoch; i <= endEpoch; i++) {
      const epochOffset = (i - startEpoch) * epochDuration;
      const phase = getPhaseForEpoch(i);
      if (phase) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: epochOffset,
          x1: epochOffset + epochDuration,
          y0: 0,
          y1: 1,
          fillcolor: PHASE_COLORS[phase],
          line: { width: 0 },
          layer: 'below',
        });
      }
    }
  }

  // Second pass: add sleep state shading and signal data
  for (let i = startEpoch; i <= endEpoch; i++) {
    const epochSignal = signal[i] || [];
    const epochOffset = (i - startEpoch) * epochDuration;

    // Add time points for this epoch
    for (let j = 0; j < epochSignal.length; j++) {
      const t = timeAxis[j] !== undefined ? timeAxis[j] : (j / epochSignal.length) * epochDuration;
      xData.push(epochOffset + t);
      yData.push(epochSignal[j]);
    }

    // Add shading for this epoch (sleep state)
    const isCurrentEpoch = i === currentEpoch;
    const score = epochs[i]?.score as SleepState || 'Unscored';
    const color = isCurrentEpoch ? SLEEP_COLORS_DARK[score] : SLEEP_COLORS_LIGHT[score];

    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: epochOffset,
      x1: epochOffset + epochDuration,
      y0: 0,
      y1: 1,
      fillcolor: color,
      line: { width: 0 },
      layer: 'below',
    });

    // Build tick labels with actual time if available
    const timestamp = getTimestampForEpoch(i);
    tickvals.push(epochOffset + epochDuration / 2); // center of epoch
    if (timestamp) {
      const hours = timestamp.getHours().toString().padStart(2, '0');
      const minutes = timestamp.getMinutes().toString().padStart(2, '0');
      const seconds = timestamp.getSeconds().toString().padStart(2, '0');
      ticktext.push(`${hours}:${minutes}:${seconds}`);
    } else {
      ticktext.push(`${epochs[i].timestamp_seconds}s`);
    }
  }

  // Calculate y-axis range
  const yMin = Math.min(...yData.filter(y => isFinite(y)));
  const yMax = Math.max(...yData.filter(y => isFinite(y)));
  const yPadding = (yMax - yMin) * 0.1;

  return (
    <Plot
      data={[
        {
          x: xData,
          y: yData,
          type: 'scattergl',
          mode: 'lines',
          line: { color: 'black', width: 1 },
          hoverinfo: 'skip',
        },
      ]}
      layout={{
        title: {
          text: title,
          font: { size: 14 },
        },
        xaxis: {
          title: { text: recordingStartTime ? 'Time' : 'Time (s)' },
          showgrid: true,
          gridcolor: '#e5e7eb',
          range: [0, (endEpoch - startEpoch + 1) * epochDuration],
          tickmode: 'array' as const,
          tickvals,
          ticktext,
        },
        yaxis: {
          title: { text: signalType === 'eeg' ? 'EEG (μV)' : 'EMG (μV)' },
          showgrid: true,
          gridcolor: '#e5e7eb',
          range: [yMin - yPadding, yMax + yPadding],
        },
        shapes: shapes as Layout['shapes'],
        margin: { l: 60, r: 20, t: 40, b: 40 },
        height,
        autosize: true,
        showlegend: false,
      }}
      config={{
        displayModeBar: false,
        responsive: true,
      }}
      style={{ width: '100%' }}
    />
  );
}
