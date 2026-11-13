# The model

What it is, what it has learned, what it gets wrong, and the decisions already made.
For *how to run* things see `docs/guides/RETRAIN_MODEL.md` and `EVALUATE_MODEL.md`.

## Task

Input: a 10-second epoch of EEG and EMG from a rat. Output: Wake, Non REM, or REM.
Ground truth: a trained human's scores for the same epochs. Roughly 85,000 labeled
epochs from 10 recordings (one 24-hour baseline day per animal, `ZZ01`–`ZZ32`).
Class balance is about 45% Wake, 47% NREM, 8% REM, so REM is the rare class and the
hard one.

## Pipeline

```
WAV ─▶ 10 s epochs ─▶ features (9 numbers per epoch) ─▶ per-recording normalization ─▶ XGBoost ─▶ state
                                                                                      └▶ signal QC → "Unscored"
                                                                                      └▶ reliability warnings
```

**Features (the 9 the served model uses).** Three EEG amplitude statistics
(`EEG_quantile_80`, `EEG_ptp`, `EEG_ss`), three EMG statistics (`EMG_std`,
`EMG_ptp`, `EMG_events` = count of large-EMG samples), and three spectral ratios
from an FFT of the epoch (`delta_rel` 0.5–4 Hz, `theta_rel` 5.5–8.5 Hz, and
`theta_over_delta`). Spectra are smoothed over neighbouring epochs before the
ratios are taken. The intuition: NREM = high delta, low EMG; REM = high theta,
very low EMG; Wake = high EMG, mixed EEG.

**Normalization.** Every animal's electrodes have different gain, so amplitude
features are scaled per recording: `(x - median) / MAD` over all epochs of that
recording (`robust_mad`). Ratios are left alone. The old method divided by a
user-chosen "baseline Wake epoch"; it still exists for the legacy served model but
is deprecated and the GUI input for it will go away once a `robust_mad` model is
served.

**Classifier.** XGBoost (`tree_method="hist"`), hyperparameters chosen by random
search with grouped time-series cross-validation. See `docs/MODEL_VALIDATION.md`
for why ordinary k-fold is wrong here.

**Rule-based filter.** A post-hoc smoother (no REM directly after Wake, no isolated
single epochs) from the original project. It is still computed and the GUI can
toggle it, but **the default view is unfiltered** because the lab counts
single-epoch REM as real and the filter deleted it. Don't re-enable by default.

**Signal QC → Unscored.** Epochs with a flat channel, clipping, non-finite values,
or an EEG amplitude far outside the recording's distribution are returned as
`Unscored` rather than guessed. Typically 0–0.3% of a clean recording.

## Reliability warnings (recording-level)

Besides scores, the API returns a `reliability` block and the GUI shows an amber
banner when any threshold is crossed. These describe the whole recording, not single
epochs. Thresholds were set on the 10-animal cohort and should be revisited as it
grows.

| Metric | Threshold | What it catches | Example |
|---|---|---|---|
| QC fail % | > 1% | Noisy recording | ZZ27 (2.1%), ZZ30 (1.7%) |
| EMG dynamic range (p05/p50 of per-epoch EMG std) | > 0.6 | EMG channel with little atonia contrast (low gain). Model will miss REM | ZZ05 (0.91) |
| OOD % (epochs beyond Mahalanobis q=0.999 from training features) | > 10% | Recording unlike the training data | ZZ05, ZZ27, ZZ30 (15–30%) |

Why recording-level: on a globally shifted animal, 15–30% of epochs cross the
per-epoch OOD line while the model still scores them at about 0.9 accuracy. Blanking
them would throw away mostly correct scores. A weak-theta electrode (ZZ01) is *not*
detectable by any of these without labels.

## Performance

Honest estimate = leave-one-animal-out (train on 8, test on the 9th, repeat).
The served model (`XGBoost_1.2.4`, v1 features, 5 animals, time-split holdout) and
the v2.1 candidate (9 animals, LOO):

| | kappa | balanced acc | REM recall | REM precision | transition acc |
|---|---|---|---|---|---|
| Served 1.2.4 (holdout) | 0.84 | – | 0.76 | – | 0.71 |
| v2.1 candidate (LOO, 9 animals) | 0.84 | 0.88 | 0.80 | 0.90 | 0.74 |

Per-animal REM recall in the candidate ranges from 0.66 (ZZ05) to 0.97 (ZZ25).
Human-vs-human agreement on REM is not much better than this, so expect the
ceiling to be label noise, not model capacity.

**Where the errors are.** REM↔NREM at bout boundaries and in single-epoch REM bouts
(recall by bout length: 1 epoch 0.15, 2–3 epochs 0.28, 4–9 epochs 0.65, 10+ epochs
0.90). Singleton REM epochs have weaker theta than sustained REM and are only
weakly separable from NREM per epoch (AUC 0.78); this is a ceiling for any
per-epoch classifier.

## Decisions already made (with reasons)

- **Unfiltered scores by default.** Lab convention: single-epoch REM counts.
- **Train on unfiltered, human-corrected labels only.** Never on `_scores.csv`
  model exports. Details in `docs/DATA_FORMATS.md`.
- **ZZ01 excluded** from training (weak-theta electrode) via `labels.json`.
- **No HMM / Viterbi smoothing.** It raises pooled REM recall to 0.85 but erases
  single-epoch REM.
- **Cost-sensitive REM weights only 1–2**, not 3–5: higher weights trade precision
  for recall the lab doesn't want.
- **OOD is a recording warning, not per-epoch blanking** (2026-11-13).
- **`robust_mad` normalization replaces the baseline epoch.** Remove the GUI input
  once a `robust_mad` model is deployed.

## What's next (open backlog, in priority order)

1. Implement floor-ceiling EMG scaling in `normalize.py` (prototype and numbers in
   `model/experiments/emg_scaling_experiment.py`; +2 points REM recall on the
   affected animals, no animal loses). Bump `FEATURE_CODE_VERSION`.
2. Wire the v2 feature set into `model/train.py` (currently only in experiments),
   train and promote the v2.1 candidate with its OOD detector.
3. Remove the baseline-epoch input from the GUI and API once (2) is served.
4. Re-review ZZ27's labels with the lab; ask how the older cohort was scored.
5. Revisit reliability thresholds when the cohort passes ~15 animals.

## Where the evidence is

- `docs/REM_RECALL_EXPERIMENT.md`: the full experiment log (v1 vs v2, class
  weights, HMM, per-animal diagnosis, EMG scaling, short-bout analysis, scorability
  metrics). Long; search for the heading you need.
- `docs/recording_scorability_metrics_2026-10-26.csv`: the label-free metrics table.
- `model_cards/` (local only, not in git): HTML reports per version including the
  v2.1 candidate card and a stakeholder walkthrough.
- `model/CHANGELOG.md`: dated log of model and data changes.
- Model metadata inside each `.pkl` (see `EVALUATE_MODEL.md`, "Reading the model's own record").
