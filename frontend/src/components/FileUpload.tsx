/**
 * File upload component with drag-and-drop support.
 *
 * React components are functions that return JSX (HTML-like syntax).
 * Props are like function arguments.
 */

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAppStore } from '../store/useAppStore';
import { scoreFile } from '../api/sleepApi';

export function FileUpload() {
  // Get state and actions from the store
  const {
    uploadState,
    uploadProgress,
    errorMessage,
    baselineEpoch,
    setUploadState,
    setUploadProgress,
    setError,
    setFileName,
    setScoringResults,
    setBaselineEpoch,
    reset,
  } = useAppStore();

  // Handle file drop/selection
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setError('Please upload a WAV file');
      return;
    }

    setFileName(file.name);
    setError(null);
    setUploadState('uploading');
    setUploadProgress(0);

    try {
      // Score the file with signal data for visualization
      const result = await scoreFile(file, baselineEpoch, true, (progress) => {
        setUploadProgress(progress);
        if (progress === 100) {
          setUploadState('processing');
        }
      });

      if (result.success) {
        setScoringResults(
          result.epochs,
          result.unfiltered_epochs,
          result.summary,
          result.model_version,
          result.samplerate,
          result.signal_data
        );
      } else {
        setError('Scoring failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  }, [baselineEpoch, setUploadState, setUploadProgress, setError, setFileName, setScoringResults]);

  // Set up dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/wav': ['.wav'],
      'audio/wave': ['.wav'],
    },
    multiple: false,
    disabled: uploadState === 'uploading' || uploadState === 'processing',
  });

  // Render different states
  const isProcessing = uploadState === 'uploading' || uploadState === 'processing';

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Baseline epoch selector */}
      <div className="mb-4 flex items-center gap-4">
        <label htmlFor="baseline-epoch" className="text-sm font-medium text-gray-700">
          Baseline Epoch:
        </label>
        <input
          id="baseline-epoch"
          type="number"
          min="0"
          value={baselineEpoch}
          onChange={(e) => setBaselineEpoch(parseInt(e.target.value) || 0)}
          className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm"
          disabled={isProcessing}
        />
        <span className="text-xs text-gray-500">
          (Choose a Wake epoch for normalization)
        </span>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
          ${uploadState === 'error' ? 'border-red-300 bg-red-50' : ''}
        `}
      >
        <input {...getInputProps()} />

        {uploadState === 'idle' && (
          <>
            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="mt-2 text-sm text-gray-600">
              {isDragActive ? 'Drop the WAV file here...' : 'Drag and drop a WAV file here, or click to select'}
            </p>
            <p className="mt-1 text-xs text-gray-500">
              Stereo WAV: Channel 1 = EEG, Channel 2 = EMG
            </p>
          </>
        )}

        {uploadState === 'uploading' && (
          <div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-600">Uploading... {uploadProgress}%</p>
          </div>
        )}

        {uploadState === 'processing' && (
          <div>
            <div className="animate-spin mx-auto h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full" />
            <p className="mt-2 text-sm text-gray-600">Processing... This may take a moment.</p>
          </div>
        )}

        {uploadState === 'error' && (
          <div>
            <p className="text-sm text-red-600">{errorMessage}</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                reset();
              }}
              className="mt-2 text-sm text-blue-600 hover:underline"
            >
              Try again
            </button>
          </div>
        )}

        {uploadState === 'complete' && (
          <div>
            <p className="text-sm text-green-600">Scoring complete!</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                reset();
              }}
              className="mt-2 text-sm text-blue-600 hover:underline"
            >
              Score another file
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
