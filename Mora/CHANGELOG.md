## TODO:
- Light/dark: allow multiple blocks for light/dark with user-specified start + end
- if user exits QInputDialog that opens from score_data(self), do not score the data!

## 2025-01-20
### Fixed
- Allow multiple light/dark periods to be applied to a single recording.

## 2025-01-07
### Fixed
- Rule based filter now applied when scoring data in `update_objects.py`

## 2024-07-21
### ADDED
- `calculate_dark_phase_from_timestamp()` in `update_objects.py`
- `dark_phase` column in export

## 2024-07-20
### ADDED
- `LightDark_Dialog()` to record beginninng and end of dark phases 
- `getDarkStartEnd` to store dark start and end times
