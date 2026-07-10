/**
 * SignalPlot - Reusable time series plot for EEG/EMG signals.
 *
 * Displays signal data with epoch shading based on sleep state.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';
import { SLEEP_COLORS_LIGHT, SLEEP_COLORS_DARK } from '../types/scoring';
import type { SleepState } from '../types/scoring';
import type { Layout } from 'plotly.js';

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
    getTimestampForEpoch,
    yAxisRanges,
    setYAxisRange,
  } = useAppStore();

  // User-set y-range for this signal, persisted across epochs (null = auto-fit).
  const manualRange = yAxisRanges[signalType];

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

  // Concatenate signals for visible epochs
  const xData: number[] = [];
  const yData: number[] = [];
  const shapes: Partial<Layout['shapes']>[number][] = [];
  const tickvals: number[] = [];
  const ticktext: string[] = [];

  // Build sleep state shading and signal data
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

  // Calculate y-axis range.
  // If the user has set a range on the axis, use it and keep it fixed as they
  // navigate epochs. Otherwise auto-fit to the visible window.
  let yRange: [number, number];
  if (manualRange) {
    yRange = manualRange;
  } else {
    const finiteY = yData.filter(y => isFinite(y));
    const yMin = Math.min(...finiteY);
    const yMax = Math.max(...finiteY);
    const yPadding = (yMax - yMin) * 0.1;
    yRange = [yMin - yPadding, yMax + yPadding];
  }

  // Capture manual axis edits (drag-zoom or typing an endpoint) so they persist.
  // Double-clicking the plot autoranges, which clears the manual range.
  // Note: editing a single endpoint fires only that one key (e.g. just
  // "yaxis.range[1]"), so merge partial edits with the current range.
  const handleRelayout = (e: Record<string, unknown>) => {
    if (e['yaxis.autorange']) {
      setYAxisRange(signalType, null);
      return;
    }

    // Plotly sometimes sends the whole range as an array under "yaxis.range".
    const arr = e['yaxis.range'];
    if (Array.isArray(arr) && arr.length === 2) {
      const lo = Number(arr[0]);
      const hi = Number(arr[1]);
      if (isFinite(lo) && isFinite(hi)) setYAxisRange(signalType, [lo, hi]);
      return;
    }

    const rawLo = e['yaxis.range[0]'];
    const rawHi = e['yaxis.range[1]'];
    if (rawLo === undefined && rawHi === undefined) return; // not a y-axis change

    const current = manualRange ?? yRange;
    const lo = rawLo !== undefined ? Number(rawLo) : current[0];
    const hi = rawHi !== undefined ? Number(rawHi) : current[1];
    if (isFinite(lo) && isFinite(hi)) setYAxisRange(signalType, [lo, hi]);
  };

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
          range: yRange,
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
        // Allow clicking the axis min/max labels to type a value; keep everything
        // else (titles, shapes, legend) locked so only the axis range is editable.
        editable: true,
        edits: {
          axisTitleText: false,
          titleText: false,
          annotationText: false,
          annotationPosition: false,
          annotationTail: false,
          legendText: false,
          legendPosition: false,
          colorbarTitleText: false,
          colorbarPosition: false,
          shapePosition: false,
        },
      }}
      onRelayout={handleRelayout}
      style={{ width: '100%' }}
    />
  );
}
