/**
 * ApiKeySettings - Simple component for entering and managing API key.
 *
 * The API key is stored in localStorage and sent with all API requests.
 * This is only needed if the backend has API key authentication enabled.
 */

import { useState, useEffect } from 'react';
import { getApiKey, setApiKey } from '../api/sleepApi';

interface ApiKeySettingsProps {
  onKeyChange?: () => void;  // Called when key is saved/cleared
}

export function ApiKeySettings({ onKeyChange }: ApiKeySettingsProps) {
  const [inputValue, setInputValue] = useState('');
  const [isKeySet, setIsKeySet] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'info'; text: string } | null>(null);

  // Load current state on mount
  useEffect(() => {
    const currentKey = getApiKey();
    setIsKeySet(!!currentKey);
    if (currentKey) {
      setInputValue(currentKey);
    }
  }, []);

  const handleSave = () => {
    const trimmedKey = inputValue.trim();
    if (!trimmedKey) {
      setMessage({ type: 'info', text: 'Please enter an API key' });
      return;
    }

    setApiKey(trimmedKey);
    setIsKeySet(true);
    setMessage({ type: 'success', text: 'API key saved! It will be used for all requests.' });
    onKeyChange?.();

    // Clear message after 3 seconds
    setTimeout(() => setMessage(null), 3000);
  };

  const handleClear = () => {
    setApiKey(null);
    setInputValue('');
    setIsKeySet(false);
    setShowKey(false);
    setMessage({ type: 'info', text: 'API key cleared. Requests will be sent without authentication.' });
    onKeyChange?.();

    setTimeout(() => setMessage(null), 3000);
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold mb-2">API Key</h3>
      <p className="text-sm text-gray-500 mb-3">
        {isKeySet
          ? "Your API key is saved and will be included in all requests."
          : "If the server requires authentication, enter your API key here."
        }
      </p>

      {/* Status indicator */}
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-2 h-2 rounded-full ${isKeySet ? 'bg-green-500' : 'bg-gray-300'}`} />
        <span className="text-sm text-gray-600">
          {isKeySet ? 'Key configured' : 'No key set (auth disabled)'}
        </span>
      </div>

      {/* Key input */}
      <div className="flex gap-2 mb-2">
        <div className="relative flex-1">
          <input
            type={showKey ? 'text' : 'password'}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter your API key"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-16"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-gray-500 hover:text-gray-700"
          >
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleSave}
          className="px-4 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
        >
          Save Key
        </button>
        {isKeySet && (
          <button
            onClick={handleClear}
            className="px-4 py-2 text-red-500 text-sm border border-red-300 rounded hover:bg-red-50 transition-colors"
          >
            Clear Key
          </button>
        )}
      </div>

      {/* Message */}
      {message && (
        <div className={`mt-3 p-2 rounded text-sm ${
          message.type === 'success'
            ? 'bg-green-50 text-green-700 border border-green-200'
            : 'bg-blue-50 text-blue-700 border border-blue-200'
        }`}>
          {message.text}
        </div>
      )}

      {/* Help text */}
      <p className="mt-3 text-xs text-gray-400">
        Your key is stored locally in your browser and never sent anywhere except the scoring API.
      </p>
    </div>
  );
}
