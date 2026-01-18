/**
 * PowerSpectrum - Frequency domain plot showing power spectrum for current epoch.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';

interface PowerSpectrumProps {
  height?: number;
}

export function PowerSpectrum({ height = 200 }: PowerSpectrumProps) {
  const { signalData, currentEpoch } = useAppStore();

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

  return (
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
      style={{ width: '100%' }}
    />
  );
}
