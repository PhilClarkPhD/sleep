/**
 * PowerSpectrum - Frequency domain plot showing power spectrum for current epoch.
 *
 * The Y-axis auto-scales to each epoch by default. "Lock Y" freezes the current
 * range so epochs can be compared on the same scale while navigating, and
 * exposes min/max boxes to type an exact range. A drag-zoom also locks it.
 * Double-click or "Unlock Y" to return to auto-scaling.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';
import { YAxisControl } from './YAxisControl';

interface PowerSpectrumProps {
  height?: number;
}

export function PowerSpectrum({ height = 200 }: PowerSpectrumProps) {
  const { signalData, currentEpoch, yAxisRanges, setYAxisRange } = useAppStore();
  const manualRange = yAxisRanges.power;

  if (!signalData || signalData.power_spectrum.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 flex items-center justify-center" style={{ height }}>
        <p className="text-gray-500">No power data</p>
      </div>
    );
  }

  const powerData = signalData.power_spectrum[currentEpoch] || [];
  const freqAxis = signalData.freq_axis;

  // Limit to 0-50 Hz for display
  const maxFreq = 50;
  const displayPower = powerData.slice(0, maxFreq * 10); // 0.1 Hz resolution
  const displayFreq = freqAxis.slice(0, maxFreq * 10);

  // Auto range for this epoch (used when not locked, and captured on Lock).
  const finite = displayPower.filter((v) => isFinite(v));
  const autoMax = finite.length ? Math.max(...finite) : 1;
  const autoRange: [number, number] = [0, autoMax * 1.05];
  const yRange = manualRange ?? autoRange;

  // Persist drag-zoom edits as a locked range; autorange (double-click) clears them.
  const handleRelayout = (e: Record<string, unknown>) => {
    if (e['yaxis.autorange']) {
      setYAxisRange('power', null);
      return;
    }
    const arr = e['yaxis.range'];
    if (Array.isArray(arr) && arr.length === 2) {
      const lo = Number(arr[0]);
      const hi = Number(arr[1]);
      if (isFinite(lo) && isFinite(hi)) setYAxisRange('power', [lo, hi]);
      return;
    }
    const rawLo = e['yaxis.range[0]'];
    const rawHi = e['yaxis.range[1]'];
    if (rawLo === undefined && rawHi === undefined) return;
    const lo = rawLo !== undefined ? Number(rawLo) : yRange[0];
    const hi = rawHi !== undefined ? Number(rawHi) : yRange[1];
    if (isFinite(lo) && isFinite(hi)) setYAxisRange('power', [lo, hi]);
  };

  return (
    <div className="relative">
      <YAxisControl plot="power" autoRange={autoRange} />
      <Plot
        data={[
          {
            x: displayFreq,
            y: displayPower,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.2)',
            line: { color: '#3b82f6', width: 2 },
            hovertemplate: '%{x:.1f} Hz<br>Power: %{y:.2e}<extra></extra>',
          },
        ]}
        layout={{
          title: {
            text: `Power Spectrum (Epoch ${currentEpoch})`,
            font: { size: 14 },
          },
          xaxis: {
            title: { text: 'Frequency (Hz)' },
            showgrid: true,
            gridcolor: '#e5e7eb',
            range: [0, maxFreq],
          },
          yaxis: {
            title: { text: 'Power' },
            showgrid: true,
            gridcolor: '#e5e7eb',
            type: 'linear',
            range: yRange,
          },
          margin: { l: 60, r: 20, t: 40, b: 40 },
          height,
          autosize: true,
          showlegend: false,
          // Highlight delta (0.5-4 Hz) and theta (5.5-8.5 Hz) bands
          shapes: [
            {
              type: 'rect',
              xref: 'x',
              yref: 'paper',
              x0: 0.5,
              x1: 4,
              y0: 0,
              y1: 1,
              fillcolor: 'rgba(59, 130, 246, 0.1)',
              line: { width: 0 },
              layer: 'below',
            },
            {
              type: 'rect',
              xref: 'x',
              yref: 'paper',
              x0: 5.5,
              x1: 8.5,
              y0: 0,
              y1: 1,
              fillcolor: 'rgba(34, 197, 94, 0.1)',
              line: { width: 0 },
              layer: 'below',
            },
          ],
          annotations: [
            {
              x: 2.25,
              y: 1,
              xref: 'x',
              yref: 'paper',
              text: 'δ',
              showarrow: false,
              font: { size: 12, color: '#3b82f6' },
            },
            {
              x: 7,
              y: 1,
              xref: 'x',
              yref: 'paper',
              text: 'θ',
              showarrow: false,
              font: { size: 12, color: '#22c55e' },
            },
          ],
        }}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
        onRelayout={handleRelayout}
        style={{ width: '100%' }}
      />
    </div>
  );
}
