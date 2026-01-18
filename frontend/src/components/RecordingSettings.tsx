/**
 * RecordingSettings - Component for setting recording start time and light/dark phases.
 *
 * Allows users to:
 * - Set the recording start date/time
 * - Define light and dark phases with start/end times
 */

import { useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import type { PhaseType } from '../types/scoring';

export function RecordingSettings() {
  const {
    recordingStartTime,
    lightDarkPhases,
    setRecordingStartTime,
    addLightDarkPhase,
    updateLightDarkPhase,
    removeLightDarkPhase,
  } = useAppStore();

  const [newPhaseType, setNewPhaseType] = useState<PhaseType>('light');
  const [newPhaseStart, setNewPhaseStart] = useState('07:00');
  const [newPhaseEnd, setNewPhaseEnd] = useState('19:00');

  // Convert ISO string to datetime-local format for input
  const formatForInput = (isoString: string | null): string => {
    if (!isoString) return '';
    const date = new Date(isoString);
    // Format: YYYY-MM-DDTHH:MM
    return date.toISOString().slice(0, 16);
  };

  // Convert datetime-local input to ISO string
  const handleStartTimeChange = (value: string) => {
    if (value) {
      setRecordingStartTime(new Date(value).toISOString());
    } else {
      setRecordingStartTime(null);
    }
  };

  const handleAddPhase = () => {
    addLightDarkPhase(newPhaseType, newPhaseStart, newPhaseEnd);
    // Toggle to the other type for convenience
    setNewPhaseType(newPhaseType === 'light' ? 'dark' : 'light');
    setNewPhaseStart(newPhaseType === 'light' ? '19:00' : '07:00');
    setNewPhaseEnd(newPhaseType === 'light' ? '07:00' : '19:00');
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-4">Recording Settings</h3>

      {/* Recording Start Time */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Recording Start Time
        </label>
        <input
          type="datetime-local"
          value={formatForInput(recordingStartTime)}
          onChange={(e) => handleStartTimeChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
        <p className="text-xs text-gray-500 mt-1">
          Set when the recording started to calculate timestamps for each epoch
        </p>
      </div>

      {/* Light/Dark Phases */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Light/Dark Phases
        </label>

        {/* Existing phases */}
        {lightDarkPhases.length > 0 && (
          <div className="space-y-2 mb-3">
            {lightDarkPhases.map((phase) => (
              <div
                key={phase.id}
                className={`flex items-center gap-2 p-2 rounded ${
                  phase.type === 'light' ? 'bg-yellow-50' : 'bg-gray-800'
                }`}
              >
                <span
                  className={`text-sm font-medium w-12 ${
                    phase.type === 'light' ? 'text-yellow-800' : 'text-gray-200'
                  }`}
                >
                  {phase.type === 'light' ? 'Light' : 'Dark'}
                </span>
                <input
                  type="time"
                  value={phase.startTime}
                  onChange={(e) =>
                    updateLightDarkPhase(phase.id, { startTime: e.target.value })
                  }
                  className="px-2 py-1 border border-gray-300 rounded text-sm"
                />
                <span className={phase.type === 'light' ? 'text-gray-600' : 'text-gray-400'}>
                  to
                </span>
                <input
                  type="time"
                  value={phase.endTime}
                  onChange={(e) =>
                    updateLightDarkPhase(phase.id, { endTime: e.target.value })
                  }
                  className="px-2 py-1 border border-gray-300 rounded text-sm"
                />
                <button
                  onClick={() => removeLightDarkPhase(phase.id)}
                  className="ml-auto text-red-500 hover:text-red-700 text-sm"
                  title="Remove phase"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Add new phase */}
        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded">
          <select
            value={newPhaseType}
            onChange={(e) => setNewPhaseType(e.target.value as PhaseType)}
            className="px-2 py-1 border border-gray-300 rounded text-sm"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
          <input
            type="time"
            value={newPhaseStart}
            onChange={(e) => setNewPhaseStart(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm"
          />
          <span className="text-gray-600">to</span>
          <input
            type="time"
            value={newPhaseEnd}
            onChange={(e) => setNewPhaseEnd(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm"
          />
          <button
            onClick={handleAddPhase}
            className="ml-auto px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
          >
            Add Phase
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Phases can span midnight (e.g., Dark: 19:00 to 07:00)
        </p>
      </div>

      {/* Quick presets */}
      <div className="border-t pt-3">
        <p className="text-sm text-gray-600 mb-2">Quick presets:</p>
        <div className="flex gap-2">
          <button
            onClick={() => {
              addLightDarkPhase('light', '07:00', '19:00');
              addLightDarkPhase('dark', '19:00', '07:00');
            }}
            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50"
          >
            12h/12h (7am-7pm)
          </button>
          <button
            onClick={() => {
              addLightDarkPhase('light', '06:00', '18:00');
              addLightDarkPhase('dark', '18:00', '06:00');
            }}
            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50"
          >
            12h/12h (6am-6pm)
          </button>
        </div>
      </div>
    </div>
  );
}
