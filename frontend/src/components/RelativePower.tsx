/**
 * RelativePower - Bar chart showing delta and theta relative power for current epoch.
 */

import Plot from 'react-plotly.js';
import { useAppStore } from '../store/useAppStore';

interface RelativePowerProps {
  height?: number;
}

export function RelativePower({ height = 200 }: RelativePowerProps) {
  const { signalData, currentEpoch } = useAppStore();

  if (!signalData) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 flex items-center justify-center" style={{ height }}>
        <p className="text-gray-500">No power data</p>
      </div>
    );
  }

  const deltaPower = signalData.delta_power[currentEpoch] || 0;
  const thetaPower = signalData.theta_power[currentEpoch] || 0;

  return (
    <Plot
      data={[
        {
          x: ['Delta', 'Theta'],
          y: [deltaPower, thetaPower],
          type: 'bar',
          marker: {
            color: ['#3b82f6', '#22c55e'],
          },
          text: [
            `${(deltaPower * 100).toFixed(1)}%`,
            `${(thetaPower * 100).toFixed(1)}%`,
          ],
          textposition: 'outside',
          hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
        },
      ]}
      layout={{
        title: {
          text: 'Relative Power',
          font: { size: 14 },
        },
        xaxis: {
          title: { text: '' },
          showgrid: false,
        },
        yaxis: {
          title: { text: 'Proportion' },
          showgrid: true,
          gridcolor: '#e5e7eb',
          range: [0, Math.max(0.5, deltaPower * 1.2, thetaPower * 1.2)],
        },
        margin: { l: 60, r: 20, t: 40, b: 40 },
        height,
        autosize: true,
        showlegend: false,
        bargap: 0.3,
      }}
      config={{
        displayModeBar: false,
        responsive: true,
      }}
      style={{ width: '100%' }}
    />
  );
}
