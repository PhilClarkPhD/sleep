/**
 * Saving files from the browser.
 *
 * Web pages cannot write to an arbitrary path on disk. There are three
 * progressively better options, and this module picks the best one available:
 *
 * 1. Plain download (every browser): the file lands in the browser's default
 *    Downloads folder with the suggested name. No choice of location.
 * 2. Save dialog (`showSaveFilePicker`, Chrome/Edge/Opera on desktop): the user
 *    picks the location and name each time.
 * 3. Export folder (`showDirectoryPicker`, same browsers): the user picks a
 *    folder once per session; later exports are written straight into it with
 *    no dialog. This is the fastest path when scoring many files.
 *
 * Safari and Firefox only support option 1.
 */

import { downloadFile } from '../api/sleepApi';

// Minimal typings for the File System Access API (not in TypeScript's DOM lib yet).
interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: { description?: string; accept: Record<string, string[]> }[];
}
interface DirectoryPickerOptions {
  id?: string;
  mode?: 'read' | 'readwrite';
  startIn?: string;
}
type PickerWindow = Window & {
  showSaveFilePicker?: (opts?: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;
  showDirectoryPicker?: (opts?: DirectoryPickerOptions) => Promise<FileSystemDirectoryHandle>;
};

const win = window as PickerWindow;

/** True when the browser lets the user choose where files are saved. */
export function canChooseSaveLocation(): boolean {
  return typeof win.showDirectoryPicker === 'function' || typeof win.showSaveFilePicker === 'function';
}

/** True when the browser supports picking an export folder. */
export function canChooseExportFolder(): boolean {
  return typeof win.showDirectoryPicker === 'function';
}

/**
 * Ask the user to pick an export folder. Returns null if they cancel or the
 * browser does not support it.
 */
export async function pickExportFolder(): Promise<FileSystemDirectoryHandle | null> {
  if (!win.showDirectoryPicker) return null;
  try {
    return await win.showDirectoryPicker({ id: 'mora-export', mode: 'readwrite' });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return null; // user cancelled
    throw err;
  }
}

async function writeToHandle(handle: FileSystemFileHandle, content: string): Promise<void> {
  const writable = await handle.createWritable();
  await writable.write(content);
  await writable.close();
}

export type SaveOutcome =
  | { method: 'folder'; filename: string; folder: string }
  | { method: 'dialog'; filename: string }
  | { method: 'download'; filename: string }
  | { method: 'cancelled' };

/**
 * Save a text file using the best available method.
 *
 * @param content   file contents
 * @param filename  suggested file name (used as-is in folder mode, overwriting)
 * @param mimeType  MIME type for the download fallback
 * @param folder    export folder chosen earlier, or null
 */
export async function saveTextFile(
  content: string,
  filename: string,
  mimeType: string,
  folder: FileSystemDirectoryHandle | null,
): Promise<SaveOutcome> {
  // 3. Export folder: write directly, no dialog.
  if (folder) {
    try {
      const fileHandle = await folder.getFileHandle(filename, { create: true });
      await writeToHandle(fileHandle, content);
      return { method: 'folder', filename, folder: folder.name };
    } catch (err) {
      // Permission may have lapsed (e.g. after a page reload). Fall through to a dialog.
      console.warn('Export folder write failed, falling back to save dialog', err);
    }
  }

  // 2. Save dialog.
  if (win.showSaveFilePicker) {
    try {
      const handle = await win.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: 'CSV file', accept: { 'text/csv': ['.csv'] } }],
      });
      await writeToHandle(handle, content);
      return { method: 'dialog', filename: handle.name };
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return { method: 'cancelled' };
      console.warn('Save dialog failed, falling back to download', err);
    }
  }

  // 1. Plain download.
  downloadFile(content, filename, mimeType);
  return { method: 'download', filename };
}
