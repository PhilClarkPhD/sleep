"""
Discover raw recordings in a (messy, nested) data directory.

A recording is a ``{ID}_{day}.wav`` file paired with a hand-scored label file
and, optionally, a legacy ``*_epoch.txt`` baseline file. The walker is
recursive and dedups by ``ID_day`` (the same recording appears under multiple
folders in the training set).

Label-file resolution
---------------------
Lab exports are not consistently named, and one folder often holds several
candidate files (raw model output, a filtered copy, a hand-corrected copy).
Resolution order, first match wins (case-insensitive):

1. An explicit override in a ``labels.json`` placed in the recording's folder
   or in the walk root: ``{"<wav stem>": "<label filename>"}``. A ``null`` value
   EXCLUDES that recording from discovery (e.g. a known-bad recording).
2. ``{stem}_scores.txt`` — the canonical convention.
3. ``*scores*10s*.csv`` / ``*10sec*scores*.csv`` — hand-corrected 10-second
   exports from the web app (these carry ``Unscored`` epochs and are the ground
   truth when present alongside a plain ``*_scores.csv`` model export).
4. ``*(corrected).txt`` then ``*(filtered).txt`` — legacy hand-scored exports.
5. ``*_scores.csv`` — plain web-app export. May be RAW MODEL OUTPUT; a warning
   is logged when this is the only candidate.

The chosen path is logged and recorded in the feature-store manifest.
"""

import fnmatch
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

LABELS_OVERRIDE_FILE = "labels.json"

# (glob pattern relative to the wav stem, is_trusted). Patterns are matched
# case-insensitively against filenames in the wav's folder. ``{stem}`` is
# substituted with the wav stem.
SCORES_PATTERNS: List[Tuple[str, bool]] = [
    ("{stem}_scores.txt", True),
    ("*scores*10s*.csv", True),
    ("*10sec*scores*.csv", True),
    ("*(corrected).txt", True),
    ("*(filtered).txt", True),
    ("*_scores.csv", False),
]


@dataclass
class Recording:
    """One discovered recording and its associated files."""

    id: str
    day: str
    wav_path: Path
    scores_path: Optional[Path] = None
    epoch_path: Optional[Path] = None

    @property
    def id_day(self) -> str:
        return f"{self.id}_{self.day}" if self.day else self.id


def parse_wav_name(stem: str) -> Tuple[str, str]:
    """Split a recording stem into (ID, day) on the last underscore.

    ``ZZ06_BL`` -> ('ZZ06', 'BL'); ``185L_BL2`` -> ('185L', 'BL2').
    """
    if "_" in stem:
        ident, day = stem.rsplit("_", 1)
    else:
        ident, day = stem, ""
    return ident, day


def _load_overrides(directory: Path) -> Dict[str, str]:
    path = directory / LABELS_OVERRIDE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}
    return {str(k): (None if v is None else str(v)) for k, v in data.items()}


EXCLUDED = object()  # sentinel returned by find_scores_file for an excluded recording


def find_scores_file(
    wav: Path, overrides: Optional[Dict[str, Optional[str]]] = None
):
    """
    Resolve the label file for ``wav`` using the module-level resolution order.

    Args:
        wav: path to the recording WAV
        overrides: root-level ``labels.json`` mapping (folder-level overrides are
            loaded here and take precedence over it)

    Returns:
        Path to the chosen label file, None if nothing matched, or the
        ``EXCLUDED`` sentinel when ``labels.json`` maps the stem to null.
    """
    directory = wav.parent
    stem = wav.stem
    files = [f for f in directory.iterdir() if f.is_file() and f != wav]
    by_lower = {f.name.lower(): f for f in files}

    merged = dict(overrides or {})
    merged.update(_load_overrides(directory))
    if stem in merged:
        target = merged[stem]
        if target is None:
            log.warning("%s: EXCLUDED via labels.json", stem)
            return EXCLUDED
        chosen = by_lower.get(target.lower()) or (
            (directory / target) if (directory / target).is_file() else None
        )
        if chosen is None:
            log.warning("labels.json names %s for %s but it does not exist", target, stem)
        else:
            log.info("%s: labels <- %s (labels.json override)", stem, chosen.name)
            return chosen

    for pattern, trusted in SCORES_PATTERNS:
        pat = pattern.replace("{stem}", stem).lower()
        matches = sorted(f for name, f in by_lower.items() if fnmatch.fnmatchcase(name, pat))
        if not matches:
            continue
        chosen = matches[0]
        if len(matches) > 1:
            log.warning(
                "%s: several files match %r (%s); using %s",
                stem, pattern, ", ".join(m.name for m in matches), chosen.name,
            )
        if trusted:
            log.info("%s: labels <- %s", stem, chosen.name)
        else:
            log.warning(
                "%s: labels <- %s (plain web-app export; may be raw model output, "
                "not hand scores — add a labels.json override if wrong)",
                stem, chosen.name,
            )
        return chosen
    return None


def discover_recordings(
    root, require_scores: bool = True
) -> List[Recording]:
    """
    Recursively find recordings under ``root``.

    Args:
        root: directory to walk
        require_scores: if True, recordings without a scores file are skipped
            (with a warning) rather than returned unlabeled.

    Returns:
        Deduplicated list of Recording, sorted by id_day for determinism.
    """
    root = Path(root)
    root_overrides = _load_overrides(root)
    recordings: List[Recording] = []
    seen: dict = {}

    for wav in sorted(root.rglob("*.wav")):
        ident, day = parse_wav_name(wav.stem)
        scores = find_scores_file(wav, root_overrides)
        if scores is EXCLUDED:
            continue
        epoch_file = wav.with_name(f"{wav.stem}_epoch.txt")
        rec = Recording(
            id=ident,
            day=day,
            wav_path=wav,
            scores_path=scores,
            epoch_path=epoch_file if epoch_file.exists() else None,
        )

        if require_scores and rec.scores_path is None:
            log.warning("No scores file for %s (%s); skipping", rec.id_day, wav)
            continue

        if rec.id_day in seen:
            log.warning(
                "Duplicate recording %s at %s; keeping %s",
                rec.id_day,
                wav,
                seen[rec.id_day].wav_path,
            )
            continue

        seen[rec.id_day] = rec
        recordings.append(rec)

    return recordings
