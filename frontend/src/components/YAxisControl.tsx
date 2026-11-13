/**
 * YAxisControl - small overlay for locking a plot's Y-axis and typing its range.
 *
 * Unlocked: the plot auto-scales to the visible data each epoch.
 * Locked: the range is frozen (captured from the current auto range) and two
 * number boxes let the user type an exact min / max. The locked range persists
 * across epoch navigation because it lives in the global store.
 */

import { useState } from 'react';
import { useAppStore, type YAxisPlot } from '../store/useAppStore';

interface YAxisControlProps {
  plot: YAxisPlot;
  autoRange: [number, number];  // range the plot would use if unlocked
}

// Compact number formatting for the input boxes (avoid 12-digit floats).
function fmt(v: number): string {
  if (!isFinite(v)) return '';
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1000 || abs < 0.01) return v.toPrecision(3);
  return String(Math.round(v * 100) / 100);
}

export function YAxisControl({ plot, autoRange }: YAxisControlProps) {
  const manualRange = useAppStore((s) => s.yAxisRanges[plot]);
  const setYAxisRange = useAppStore((s) => s.setYAxisRange);
  const locked = manualRange !== null;

  // While the user is typing we show their draft; otherwise the boxes mirror
  // the stored range (so a drag-zoom updates them automatically).
  const [draft, setDraft] = useState<{ min: string; max: string } | null>(null);
  const minText = draft ? draft.min : manualRange ? fmt(manualRange[0]) : '';
  const maxText = draft ? draft.max : manualRange ? fmt(manualRange[1]) : '';

  const toggle = () => {
    setDraft(null);
    setYAxisRange(plot, locked ? null : autoRange);
  };

  const commit = () => {
    if (!draft) return;
    const lo = Number(draft.min);
    const hi = Number(draft.max);
    if (isFinite(lo) && isFinite(hi) && lo < hi) {
      setYAxisRange(plot, [lo, hi]);
    }
    setDraft(null); // invalid entry snaps back to the stored range
  };

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
    if (e.key === 'Escape') {
      setDraft(null);
      (e.target as HTMLInputElement).blur();
    }
  };

  const inputClass =
    'w-20 px-1 py-0.5 text-xs border border-gray-300 rounded bg-white focus:outline-none focus:ring-1 focus:ring-blue-400';

  return (
    <div className="absolute right-2 top-1 z-10 flex items-center gap-1 text-xs">
      {locked && (
        <>
          <input
            type="number"
            step="any"
            value={minText}
            onChange={(e) => setDraft({ min: e.target.value, max: maxText })}
            onBlur={commit}
            onKeyDown={onKey}
            className={inputClass}
            title="Y-axis minimum (Enter to apply)"
            aria-label={`${plot} Y-axis minimum`}
          />
          <span className="text-gray-500">to</span>
          <input
            type="number"
            step="any"
            value={maxText}
            onChange={(e) => setDraft({ min: minText, max: e.target.value })}
            onBlur={commit}
            onKeyDown={onKey}
            className={inputClass}
            title="Y-axis maximum (Enter to apply)"
            aria-label={`${plot} Y-axis maximum`}
          />
        </>
      )}
      <button
        onClick={toggle}
        className={`px-2 py-0.5 rounded border ${
          locked
            ? 'bg-blue-50 border-blue-300 text-blue-700'
            : 'bg-white border-gray-300 text-gray-600 hover:bg-gray-50'
        }`}
        title={
          locked
            ? 'Y-axis is locked; click to return to auto-scaling'
            : 'Lock the Y-axis at its current range (then type an exact min/max)'
        }
      >
        {locked ? 'Unlock Y' : 'Lock Y'}
      </button>
    </div>
  );
}
