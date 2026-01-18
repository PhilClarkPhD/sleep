/**
 * Main application component.
 *
 * This is the root component that composes all other components.
 */

import { useEffect, useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { Hypnogram } from './components/Hypnogram';
import { ScoringResults } from './components/ScoringResults';
import { EpochViewer } from './components/EpochViewer';
import { ExportPanel } from './components/ExportPanel';
import { useAppStore } from './store/useAppStore';
import { checkHealth, getModelInfo } from './api/sleepApi';

function App() {
  const { uploadState, setModelInfo, modelInfo, signalData } = useAppStore();
  const [apiStatus, setApiStatus] = useState<'checking' | 'healthy' | 'error'>('checking');
  const [apiError, setApiError] = useState<string | null>(null);

  // Check API health on mount
  useEffect(() => {
    async function init() {
      try {
        const health = await checkHealth();
        if (health.model_loaded) {
          setApiStatus('healthy');
          const info = await getModelInfo();
          setModelInfo(info);
        } else {
          setApiStatus('error');
          setApiError('Model not loaded on server');
        }
      } catch (err) {
        setApiStatus('error');
        setApiError(err instanceof Error ? err.message : 'Failed to connect to API');
      }
    }
    init();
  }, [setModelInfo]);

  const showResults = uploadState === 'complete';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Mora Sleep Scoring</h1>
            <p className="text-sm text-gray-500">Automated EEG/EMG sleep state classification</p>
          </div>
          {modelInfo && (
            <div className="text-right text-sm text-gray-500">
              <div>Model: {modelInfo.model_name}</div>
              <div>Version: {modelInfo.model_version}</div>
            </div>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* API status banner */}
        {apiStatus === 'checking' && (
          <div className="mb-6 p-4 bg-blue-50 text-blue-700 rounded-lg flex items-center gap-2">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full" />
            Connecting to API...
          </div>
        )}

        {apiStatus === 'error' && (
          <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg">
            <strong>API Error:</strong> {apiError}
            <p className="text-sm mt-1">
              Make sure the backend is running: <code className="bg-red-100 px-1 rounded">uvicorn app.main:app --port 8000</code>
            </p>
          </div>
        )}

        {/* File upload section - always visible but compact when results shown */}
        <section className={showResults ? 'mb-4' : 'mb-8'}>
          <FileUpload />
        </section>

        {/* Results section - only show when scoring is complete */}
        {showResults && (
          <div className="space-y-6">
            {/* Epoch viewer with signal plots (only if signal data available) */}
            {signalData && (
              <EpochViewer />
            )}

            {/* Hypnogram - full width */}
            <Hypnogram />

            {/* Summary statistics and export side by side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Summary statistics */}
              <ScoringResults />

              {/* Import/Export panel */}
              <ExportPanel />
            </div>

            {/* Instructions */}
            <div className="bg-white rounded-lg shadow p-4 text-sm text-gray-600">
              <h3 className="font-semibold text-gray-900 mb-2">Keyboard Shortcuts</h3>
              <ul className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">←</kbd> Previous epoch</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">→</kbd> Next epoch</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">W</kbd> Score as Wake</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">E</kbd> Score as Non REM</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">R</kbd> Score as REM</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">T</kbd> Score as Unscored</li>
                <li><kbd className="px-2 py-1 bg-gray-100 rounded">Ctrl+F</kbd> Find epoch</li>
              </ul>
              <p className="mt-3 text-xs text-gray-500">
                Each epoch is 10 seconds. Click on the hypnogram to jump to any epoch.
                Changes are auto-advanced after scoring.
              </p>
            </div>
          </div>
        )}

        {/* Instructions when idle */}
        {uploadState === 'idle' && apiStatus === 'healthy' && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">How to Use</h2>
            <ol className="list-decimal list-inside space-y-2 text-gray-600">
              <li>
                <strong>Prepare your data:</strong> Record EEG and EMG as a stereo WAV file
                (EEG on channel 1, EMG on channel 2)
              </li>
              <li>
                <strong>Set baseline epoch:</strong> Choose an epoch where the animal is awake
                (default is epoch 2, which is seconds 20-30)
              </li>
              <li>
                <strong>Upload:</strong> Drag and drop your WAV file or click to browse
              </li>
              <li>
                <strong>Review:</strong> Use the epoch viewer to inspect EEG/EMG signals and
                manually correct any mis-scored epochs
              </li>
              <li>
                <strong>Navigate:</strong> Use arrow keys or click on the hypnogram to move
                between epochs
              </li>
              <li>
                <strong>Score:</strong> Press W (Wake), E (NREM), R (REM), or T (Unscored)
                to manually score epochs
              </li>
              <li>
                <strong>Export:</strong> Download the scores as a CSV file when done
              </li>
            </ol>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-sm text-gray-500">
          Mora Sleep Scoring &middot; XGBoost-based automated sleep classification
        </div>
      </footer>
    </div>
  );
}

export default App;
