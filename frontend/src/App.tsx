/**
 * Main application component.
 *
 * This is the root component that composes all other components.
 */

import { useEffect, useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { Hypnogram } from './components/Hypnogram';
import { ScoringResults } from './components/ScoringResults';
import { ExportButton } from './components/ExportButton';
import { useAppStore } from './store/useAppStore';
import { checkHealth, getModelInfo } from './api/sleepApi';

function App() {
  const { uploadState, setModelInfo, modelInfo } = useAppStore();
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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-5xl mx-auto px-4 py-4 flex justify-between items-center">
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
      <main className="max-w-5xl mx-auto px-4 py-8">
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

        {/* File upload section */}
        <section className="mb-8">
          <FileUpload />
        </section>

        {/* Results section - only show when scoring is complete */}
        {uploadState === 'complete' && (
          <div className="space-y-6">
            {/* Export button */}
            <div className="flex justify-end">
              <ExportButton />
            </div>

            {/* Summary statistics */}
            <ScoringResults />

            {/* Hypnogram */}
            <Hypnogram />

            {/* Instructions */}
            <div className="bg-white rounded-lg shadow p-4 text-sm text-gray-600">
              <h3 className="font-semibold text-gray-900 mb-2">About the Results</h3>
              <ul className="list-disc list-inside space-y-1">
                <li>Each epoch is 10 seconds of recording</li>
                <li>Sleep states: <strong>Wake</strong> (awake), <strong>NREM</strong> (Non-REM sleep), <strong>REM</strong> (REM sleep)</li>
                <li>The baseline epoch is used for normalizing EEG/EMG features</li>
                <li>Click "Export CSV" to download the epoch-by-epoch scores</li>
              </ul>
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
                <strong>Review:</strong> Check the hypnogram and summary statistics
              </li>
              <li>
                <strong>Export:</strong> Download the scores as a CSV file
              </li>
            </ol>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-12">
        <div className="max-w-5xl mx-auto px-4 py-4 text-center text-sm text-gray-500">
          Mora Sleep Scoring &middot; XGBoost-based automated sleep classification
        </div>
      </footer>
    </div>
  );
}

export default App;
