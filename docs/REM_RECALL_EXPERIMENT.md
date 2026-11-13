interact# Improving REM recall — experiment log & results

**Goal:** raise REM recall for the sleep-scoring model without leakage and with an
honest estimate of performance on *new animals*.

**Dataset:** `/Users/phil/Downloads/Filtered Scores` — 5 rats (ZZ01, ZZ05, ZZ16,
ZZ18, ZZ19), 24 h each, 10 s epochs, ~40k scored epochs after dropping `Unscored`.
Class balance ≈ Wake 54% / NREM 38% / **REM 8%** (the hard minority).

---

## Evaluation protocol (the honest part)

- **Leave-one-animal-out (LOO)** is the only thing that produces a reported
  number. Train on 4 rats, predict the 5th, pool all 5 held-out animals, score
  once. No animal is ever in both train and test → this is a genuine
  "score a brand-new rat" estimate.
- **Nested LOO for any tunable knob.** The cost-sensitive REM weight / probability
  multiplier is chosen by an *inner* LOO over the 4 training animals only, then
  applied blind to the held-out outer animal. The test animal never influences
  its own operating point → no selection leakage.
- **Leakage controls:** `robust_mad` normalization is per-recording (within
  animal); all engineered features are label-free and computed within a recording;
  the HMM transition matrix is estimated from training animals only, per outer fold.
- **Guardrails reported every run** (recall alone is gameable): REM recall ↑ **and**
  REM precision ≥ 0.85 floor **and** Cohen's κ not worse. Plus per-class confusion
  and per-animal breakdown.

> Note: these LOO κ values (0.75–0.80) are *lower* than the ~0.84 in earlier
> project notes because that number was a within-animal temporal holdout (shares
> subjects across train/test). LOO is the more conservative, more truthful number.

---

## Steps taken

1. **Data wiring.** The label files didn't match `discovery.py`'s expected
   `{wav_stem}_scores.txt` pattern. Copied (not moved) each `*(filtered).txt`
   (ZZ16: `*(corrected).txt`) to the expected name; originals untouched. All 5
   recordings then discovered and parsed.
2. **Feature stores.** Built committed, versioned v1 (12 feat) and v2 (28 feat)
   stores with `robust_mad` normalization via `python -m sleep_features build`.
3. **v1 vs v2 baseline (LOO).** `model/compare_feature_sets.py`.
4. **Phase 0 — headroom.** REM one-vs-rest PR curve from LOO probabilities.
5. **Phase 1 — cost-sensitive operating point** (nested LOO): REM sample-weight +
   REM probability-multiplier. `model/rem_recall_experiment.py`.
6. **Phase 2 — REM-targeted features** (LOO ablation).
7. **Phase 3 — HMM/Viterbi sequence decoding** (LOO). Phases 2–3 in
   `model/rem_phase23_experiment.py`.

---

## Results

### v1 vs v2 (LOO, fixed identical hyperparameters)

| metric | v1 (12) | v2 (28) |
|---|---|---|
| Cohen κ | 0.746 | **0.791** |
| REM recall | 0.599 | **0.648** |
| REM precision | 0.878 | **0.916** |
| balanced acc | 0.789 | **0.820** |

v2 (adds sigma/beta/gamma bands, spectral entropy/edge, Hjorth params, temporal
roll/diff) wins across the board → the working baseline.

### Phase 0 — headroom (raw REM proba, pre-filter)

REM average precision 0.840. Recall reachable **by threshold alone**:
`@prec≥0.90 → 0.670`, `@0.85 → 0.733`, `@0.80 → 0.768`. Baseline argmax sat at
recall 0.648 / precision 0.916 — conservative, room to move.

### Phases 1–3 (all LOO; REM recall is the target)

| pipeline | REM rec | REM prec | κ | bal acc | macro-F1 | trans acc |
|---|---|---|---|---|---|---|
| baseline (v2, w1, rule) | 0.648 | 0.916 | 0.791 | 0.820 | 0.848 | 0.679 |
| **Phase 1** (v2, cost-sens, rule) | 0.759 | 0.853 | 0.799 | 0.853 | 0.863 | 0.683 |
| Phase 2 feats (v2+extra, w1, rule) | 0.689 | 0.909 | 0.803 | 0.835 | 0.859 | 0.680 |
| **Phase 2+cost-sens (rule)** | **0.747** | **0.870** | **0.802** | 0.849 | **0.864** | 0.677 |
| Phase 3 HMM (v2, w1) | 0.835 | 0.755 | 0.798 | 0.874 | 0.859 | 0.640 |
| Phase 2+3 HMM (v2+extra, w1) | 0.839 | 0.739 | 0.797 | 0.874 | 0.857 | 0.629 |
| cost-sens + HMM (stacked) | 0.869 | 0.583 | 0.769 | 0.867 | 0.821 | 0.588 |

New features added in Phase 2 (all label-free, within-recording): `theta_dominance`,
`theta_over_sigma`, `emg_atonia` (EMG vs recording's own p90), and bout-scale
rolling means (roll5/roll9) of theta/delta, EMG, delta, theta-dominance.

---

## Findings

- **Phase 2 features are a free win:** +0.04 REM recall, +0.01 κ, essentially no
  precision cost. Keep them permanently.
- **Cost-sensitive weighting** buys REM recall along the precision frontier and
  respects the 0.85 precision floor when the floor gates selection.
- **HMM is the single biggest REM-recall lever** (bout recovery): +0.19 recall
  (0.648 → 0.835) at w=1 — but it **trades precision to ~0.75 and lowers
  transition accuracy** (it smears the exact epoch of state changes while getting
  the bout right).
- **Do not stack cost-sensitive + HMM:** both push REM, precision collapses to
  0.58 and κ drops. The precision guardrail correctly flags this as unacceptable
  (inner selection ran to w=5 because no config met the 0.85 floor).

## Decision: drop the legacy rule-based filter

The hand-written `apply_rule_based_filter` includes a rule that converts
`Wake, Wake, REM → Wake` (filters.py) — it actively **deletes** valid REM epochs,
exactly counter to this project's goal, and it also lowered transition accuracy.
Decision (Phil): **do not apply it**; report raw per-epoch model output. All tables
above include the filter; the honest **no-filter** numbers are below.

### No-filter results (leave-one-animal-out)

| model | REM rec | REM prec | κ | bal acc | acc |
|---|---|---|---|---|---|
| v1 (12 feat) | 0.520 | 0.789 | 0.712 | 0.752 | 0.842 |
| v2 (28 feat, w1) | 0.706 | 0.863 | 0.782 | 0.831 | 0.880 |
| **v2 + Phase 2 + cost-sensitive** | **0.730** | **0.865** | **0.795** | **0.842** | **0.887** |

Removing the filter *raised* v2 REM recall (0.648 → 0.706) and transition accuracy
(0.680 → 0.728). Per-animal REM recall (recommended, no filter): ZZ01 0.55, ZZ05 0.56,
ZZ19 0.84, ZZ18 0.86, ZZ16 0.91 — the 55–91% spread is the case for more animals.

## Recommended pipeline (no filter)

**v2 + Phase 2 features + cost-sensitive weighting, raw per-epoch output.**
REM recall **0.520 → 0.730** vs the prior 12-feature model, with REM precision *also*
up (0.789 → 0.865) and κ 0.712 → 0.795.

- **Higher REM sensitivity option (future): + HMM/Viterbi** pushes REM recall to
  ~0.84 but is a learned *sequence smoother* (changes transition timing) — hold
  until more animals validate it. Not the ad-hoc rule filter.

## Per-animal REM diagnosis — the two low performers fail differently

REM recall is bimodal across animals (ZZ16/18/19 ≈ 0.84–0.91; ZZ01/ZZ05 ≈ 0.55).
Digging in (feature-store separability + raw-signal confirmation on the WAVs):

**ZZ05 — fixable by more data (distribution shift).** Its REM is the *cleanest* of all:
raw REM theta-peak prominence 8.0 and theta/delta 4.5 (highest in the cohort), within-animal
theta AUC 0.87. But it sits ~2 SD off the other four animals (atypically strong theta, 10.5%
REM vs ~7%). LOO trains on the other 4 and never learned an animal in ZZ05's regime → the
boundary misplaces it. **More animals directly fixes this.**

**ZZ01 — not fixable by more data (weak theta signal).** Its REM theta is genuinely muted:
raw theta-peak prominence 3.6 (~half the others' 7–8) and theta/delta 1.4 (vs 2.6–4.5).
Within-animal theta AUC only 0.72; missed REM goes mostly to Wake. **Confirmed from the raw
signal that this is NOT noise** — QC ~0%, HF-noise ratio 0.093 and 60 Hz line 0.39 are in line
with the clean animals. The discriminative feature (theta *amplitude*) is simply weak, pointing
to **electrode montage / anatomy**, not signal quality. More recordings won't rescue ZZ01;
checking its EEG lead placement, or adding theta-amplitude-independent features (theta
frequency/coherence, phasic EMG twitches, sequence context), might.

Raw-signal metrics (median per epoch; ZZ16 = clean high-performer reference):

| animal | REM recall | QC flagged | HF-noise (20–45 Hz) | 60 Hz line | REM θ-peak prominence | REM θ/δ (raw) |
|---|---|---|---|---|---|---|
| ZZ01 | 0.55 | 0.02% | 0.093 | 0.39 | **3.6** | **1.4** |
| ZZ05 | 0.56 | 0.00% | 0.085 | 0.44 | 8.0 | 4.5 |
| ZZ16 | 0.84 | 0.02% | 0.084 | 0.29 | 7.1 | 2.6 |

## Caveats

- **N = 5 animals.** LOO is high variance; nested inner folds are only 4 animals,
  so selected operating points are noisy and all deltas have wide error bars.
  **More scored recordings would help more than any modeling trick.**
- HMM's transition-accuracy drop means precise sleep-architecture timing
  (latencies, exact transition counts) degrades even as bout detection improves.
- A tempered HMM (transition "stickiness" / REM penalty) could sit between the two
  recommended points — a Phase 4 if a middle operating point is wanted.

## Artifacts (reusable, committed to `model/`)

- `compare_feature_sets.py` — LOO v1-vs-v2 comparison.
- `rem_recall_experiment.py` — Phase 0 headroom + Phase 1 nested cost-sensitive.
- `rem_phase23_experiment.py` — Phase 2 features + Phase 3 HMM/Viterbi.

---

# Update 2026-10-25 — 10 animals

**Dataset:** `training_data/Filtered Scores 2/` — the original 5 (ZZ01, ZZ05, ZZ16, ZZ18, ZZ19)
plus 5 new (ZZ25, ZZ27, ZZ29, ZZ30, ZZ32), ~81k scored epochs. Feature stores
`feature_store/features_2026-10-25_22-50-27.csv` (v2) / `..._22-51-28.csv` (v1), robust_mad.
Label files chosen by `sleep_features.discovery.find_scores_file` (new animals: the
`(10s)`/`10sec` hand-corrected exports; see `model/CHANGELOG.md`).

## Pooled LOO results (same protocol as above)

| pipeline | REM rec | REM prec | κ | bal acc | trans |
|---|---|---|---|---|---|
| v1 (12 feats), rule | 0.70 | | 0.795 | 0.835 | 0.669 |
| v2 (28), rule | 0.733 | 0.922 | 0.821 | 0.856 | 0.677 |
| v2, no filter | 0.753 | 0.891 | 0.823 | 0.860 | 0.743 |
| v2 + REM feats, no filter | 0.763 | 0.901 | 0.824 | 0.863 | 0.729 |
| v2 + HMM | 0.851 | 0.802 | 0.828 | 0.891 | 0.676 |
| **v2 + REM feats + nested cost-sensitive, no filter** (recommended) | **0.789** | **0.876** | **0.826** | **0.871** | **0.731** |

July (5 animals), same recommended pipeline: REM rec 0.73 / prec 0.865 / κ 0.795 / bal acc 0.842.
Selected REM weights are now 1–2 per fold (were 3–5): the unweighted model already sits at the
precision floor, so weighting buys ~2.5 pts recall for ~2.5 pts precision.

**Doubling the animals was worth as much as all of the July feature work:** the old 12-feature
model went κ 0.71 → 0.80 from data alone.

Per-animal REM recall, recommended pipeline (July → now): ZZ01 55→71, ZZ05 56→61, ZZ16 91→88,
ZZ18 86→88, ZZ19 84→79; new: ZZ25 97, ZZ27 66, ZZ29 83, ZZ30 78, ZZ32 85.

## Per-animal diagnosis (`python -m model.diagnose_animals`)

Unweighted no-filter LOO predictions; raw metrics from the WAVs.

**ZZ05 — EMG calibration, not EEG (July's diagnosis was half right).** With 10 animals its REM
theta is no longer an outlier (z ≈ 1.1 vs ~2 SD in July) yet recall barely moved (56→59→61).
Misses go to *Wake* (223 vs 126 to NREM) with median p(REM) = 0.04 (only 17% ≥ 0.2): the model
is confident, so no threshold/weighting fix. Cause: (a) REM-epoch `EMG_std` z = **+3.5**, the
largest shift in the table — raw Wake EMG std 2.5 is the lowest gain in the cohort (others
4–20) and EMG atonia REM/Wake = 0.51 (others 0.01–0.34), so after per-recording normalization
its REM EMG sits where other animals are not atonic; (b) its waking EEG is theta-rich (raw Wake
θ/δ 0.92, highest), and EEG-only REM-vs-Wake AUCs are among the lowest (γ 0.70, β 0.57,
entropy 0.69), so theta cannot rescue the Wake/REM split. Within-animal EMG AUC is fine (0.93) →
it is a cross-animal *scale* problem. **Fix direction:** EMG normalization anchored to the
recording's own quiet floor / per-recording quantile normalization of EMG features (emg_atonia,
referenced to p90, already halves the shift: z 1.9). Testable without new data.

**ZZ01 — weak-theta recording, confirmed and now the clear theta outlier** (θ-dominance z −3.2).
Raw REM θ/δ 0.90 vs NREM 0.78 — almost no REM/NREM theta contrast; REM θ-peak prominence 9.4
(lowest; cohort 10–22). Misses go to NREM and are borderline (median p(REM) 0.12, 39% ≥ 0.2),
which is why weighting lifted it 62→71 at the cost of precision 0.74→0.71. Clean recording
(QC 0.02%) but highest HF noise (0.24). Curiosity: within ZZ01, `spectral_entropy` separates
REM at AUC 0.91 (best of any animal) — its REM is distinguishable by complexity, not theta,
which a theta-trained cohort model cannot exploit. Still: electrode-placement check, not data.

**ZZ27 — labels first, signal second.** Its two score files disagree on 21% of epochs; 60% of
its REM epochs (422/701) are disputed, and the model calls REM on only 50% of those vs 80% of
undisputed REM. Labels are fragmented: 160 bouts, median 3 epochs, 33% singletons, 19% entered
directly from Wake; singleton recall is 6%. The recording is also the noisiest (QC flagged 2.2%,
10× typical; HF noise 0.21), REM θ prominence second lowest (10.0), and REM EMG is *not* below
NREM (REM/NREM 1.10; EMG REM-vs-rest AUC 0.69, lowest). **Action:** re-review ZZ27's short REM
bouts before trusting it as training/eval data; check its EMG lead.

**Cohort-level finding — the two label sources have different conventions.** Old-cohort labels
(`*(filtered).txt`) contain 0% singleton REM bouts and 0% REM entered from Wake — the legacy
rule filter was applied to them. New-cohort labels (web-app corrected exports) have 14–33%
singletons and 16–43% REM-from-Wake. Across the new animals, recall on *undisputed* REM is
0.88–1.00 (ZZ27: 0.80); nearly all the shortfall is short/disputed bouts. The training targets
are therefore stylistically inconsistent. Decide one convention (either apply a bout-plausibility
cleanup to the new labels, or use the unfiltered originals of the old labels if `ZZ01_BL.txt`
etc. are those) before the next training run.

## Updated next steps
1. Label convention decision (above) + ZZ27 review.
2. EMG scale-robust normalization experiment (targets ZZ05-type animals).
3. Wire recommended pipeline into `train.py`, ship a v2 artifact (robust_mad, no filter,
   `rem_weight` ≈ 2), and drop the baseline-epoch input from UI/API.
4. HMM as an optional "smooth" mode (recall 0.85 at precision 0.80) once labels are consistent.

## EMG scaling experiment (2026-10-25, `model/emg_scaling_experiment.py`)

Question: ZZ05's REM-epoch EMG sits +3.5 SD from the cohort after per-recording robust_mad
(affine) normalization because its EMG dynamic range is compressed (low gain / noise floor).
Does a shape-aligning per-recording transform of the EMG amplitude features fix it?

Pooled LOO, v2 + REM extras, unweighted, no filter:

| EMG scaling | REM rec | REM prec | κ | bal acc | ZZ05 EMG z | ZZ05 rec | ZZ27 rec | ZZ01 rec |
|---|---|---|---|---|---|---|---|---|
| robust_mad (current) | 0.763 | 0.901 | 0.824 | 0.863 | 3.5 | 0.59 | 0.62 | 0.62 |
| **floor-ceiling p05/p95** | **0.780** | 0.885 | **0.830** | **0.871** | 2.0 | 0.64 | 0.70 | 0.61 |
| floor-ceiling p10/p90 | 0.776 | 0.884 | 0.829 | | | 0.65 | 0.69 | 0.59 |
| floor-ceiling EMG + EEG amplitude | 0.784 | 0.884 | 0.828 | 0.871 | | 0.63 | 0.70 | 0.68 |
| rank (percentile) | 0.757 | 0.882 | 0.826 | 0.863 | 0.6 | 0.58 | 0.58 | 0.57 |

- **Floor-ceiling (x − p05)/(p95 − p05) on EMG_std/EMG_ptp is a free, modest win**: κ, balanced
  accuracy and REM recall all up, no animal loses > 2 pts; ZZ05 +5, ZZ27 +8. Anchor choice
  (p05/p95 vs p10/p90) does not matter. Scaling EEG amplitude features too helps ZZ01 (+7) but
  not the pool. **Recommended for the v2 pipeline.**
- **Rank normalization is the informative failure**: it aligns ZZ05's EMG almost perfectly
  (z 0.6) yet ZZ05 recall does not move, and it costs ZZ01/ZZ27 (destroys magnitude info).
  So EMG calibration is only *part* of ZZ05's problem.
- After the EMG fix ZZ05's misses split Wake 140 / NREM 164 (was 223/126), still confident
  (median p(REM) 0.05). Recall by bout length: 2–3 epochs 0.05, 4–9 0.25, **10+ 0.83**. ZZ05's
  long bouts are fine; its deficit is that 62 of its 94 REM bouts are short, and short bouts
  are missed on every animal (cohort singleton recall 6–50%, 2–3-epoch 0.1–0.7).
- **Rolling-window smearing hypothesis rejected.** Dropping the centered roll9 / roll5 / all
  temporal-context features (floor-ceiling EMG, LOO): short-bout recall is unchanged
  (singletons 0.13–0.21, 2–3 epochs 0.26–0.29 in every variant) while pooled recall falls
  0.780 → 0.767 → 0.761 → 0.705 and ZZ05 does not improve (0.64 → 0.62 → 0.63 → 0.56).
  Temporal context helps long bouts without hurting short ones; short bouts are missed
  because their per-epoch signal is genuinely transitional, not because context erases it.
  Remaining levers for short bouts are sequence-aware decoding (HMM reached 0.85 pooled
  recall) or more short-bout examples under a consistent labeling convention.

## Short REM bouts: convention vs signal, and local-contrast features (2026-10-26)

Lab position: a single 10 s REM epoch counts as REM. Two questions follow.

**Is the old/new labeling-convention mismatch why short bouts are missed?** Only ~5 pts of it.
LOO over the 5 new animals, varying the training pool (floor-ceiling EMG):

| training pool | singleton rec | 2–3 rec | 10+ rec | overall |
|---|---|---|---|---|
| all 10 | 0.15 | 0.29 | 0.94 | 0.82 |
| 5 new only (consistent convention) | 0.20 | 0.35 | 0.88 | 0.79 |
| 5 old only (no singletons in labels) | 0.09 | 0.21 | 0.90 | 0.76 |

**The dominant effect is a weaker per-epoch signature.** New-animal medians: theta/delta
singleton REM 1.40 · 2–3 ep 1.95 · 4–9 ep 2.46 · 10+ ep 2.73 · NREM 0.70 · Wake 0.71; EMG is fully
atonic in all REM categories. Singletons sit midway between NREM and sustained REM.

**Local-contrast features (`model/local_contrast_experiment.py`) — null result.**
`log(col / trailing_median(col, prev W epochs))` for theta/delta, theta dominance, theta_rel,
W = 30 and 90 epochs, added to or replacing the centered roll5/roll9 means: pooled REM recall
0.776–0.782 (baseline 0.780), singleton recall 0.14–0.16 (baseline 0.15), 2–3-epoch 0.25–0.30
(0.28). Replacing the rolling means with local contrast helped ZZ01 (+6) and ZZ27 (+3) but cost
ZZ30 (−9). Not adopted.

**Implication:** HMM / sequence smoothing is the wrong tool for this lab (it erases singletons
by design). Short-bout recall is bounded by per-epoch evidence; the remaining levers are more
singleton examples under the new convention and, possibly, re-scoring the old cohort.

## ZZ01 excluded — 9-animal cohort (2026-10-26)

Phil's decision: ZZ01 is a known-bad recording (weak theta, electrode placement) and is removed from
training and evaluation. Mechanism: `training_data/Filtered Scores 2/labels.json` maps
`"ZZ01_BLSel24hr": null` → `discover_recordings` skips it (logged as EXCLUDED), so future feature-store
builds omit it. Today's numbers filter the existing 10-animal store by ID.

**v2.1 candidate on 9 animals** (v2 + REM extras, floor-ceiling EMG scaling, unweighted, no filter,
pooled LOO, 73,256 scored epochs, REM 8.2%):

| | 10 animals | **9 animals (no ZZ01)** |
|---|---|---|
| accuracy | 0.905 | **0.911** |
| Cohen's κ | 0.830 | **0.842** |
| balanced accuracy | 0.871 | **0.880** |
| REM recall / precision | 0.780 / 0.885 | **0.796 / 0.898** |
| Wake recall · NREM recall | 0.916 · 0.916 | 0.923 · 0.920 |
| transition accuracy | 0.737 | 0.740 |
| recall by bout length (1 / 2–3 / 4–9 / 10+) | .15 / .28 / .62 / .88 | .15 / .27 / .66 / .90 |

Per-animal REM recall (9): ZZ05 0.66 · ZZ16 0.84 · ZZ18 0.86 · ZZ19 0.77 · ZZ25 0.97 · ZZ27 0.69 ·
ZZ29 0.83 · ZZ30 0.77 · ZZ32 0.82. Same pipeline with MAD scaling on 9 animals: κ 0.829, REM recall
0.768 — the floor-ceiling EMG gain is larger without ZZ01 (ZZ27 0.56 → 0.69, ZZ05 0.59 → 0.66).
Stakeholder card: `model_cards/SleepScoring_v2.1_candidate.html`.

## Can we flag hard-to-score recordings before scoring? (2026-10-26, label-free)

Recording-level metrics computed from the raw WAV / feature store with NO labels, vs v2.1 REM recall
(10 animals incl. ZZ01). Raw table: `docs/recording_scorability_metrics_2026-10-26.csv`.

| metric | ZZ01 | ZZ05 | ZZ27 | ZZ30 | clean 6 (range) | verdict |
|---|---|---|---|---|---|---|
| QC fail % (per-epoch signal QC) | 0.02 | 0.00 | **2.14** | **1.73** | 0.00–0.26 | flags noisy recordings (ZZ27; ZZ30 half-hit) |
| EMG dynamic range p05/p50 (quiet vs typical epoch) | 0.69 | **0.91** | 0.52 | 0.01 | 0.06–0.46 | flags ZZ05 clearly (no atonia contrast) |
| OOD % (Mahalanobis vs other animals, q=0.999) | 6.6 | **15.0** | **29.7** | **21.4** | 0.7–11.4 | best single predictor (Spearman −0.59); flags ZZ05, ZZ27, ZZ30 |
| theta contrast, all epochs p95/p50 | 3.40 | 4.71 | 5.73 | 5.61 | 3.6–5.8 | useless (confounded by wake theta) |
| theta contrast within low-EMG epochs p90/p50 | 2.47 | 6.10 | 2.37 | 3.63 | 2.6–6.2 | weak: ZZ01/ZZ27 lowest but ZZ19/ZZ16 close |
| HF noise median | 0.093 | 0.085 | 0.092 | 0.098 | 0.08–0.12 | useless |

**Conclusion:** two of the three problem animals are detectable before scoring: ZZ05 via EMG dynamic
range, ZZ27 via QC-fail rate and OOD fraction. A weak-theta recording like ZZ01 is *not* reliably
detectable label-free with these metrics (clean, in-distribution; its deficit only shows against labels).
Proposed recording-level warning (conservative thresholds, n=10 so revisit): QC fail > 1% → "noisy";
EMG p05/p50 > 0.6 → "weak EMG atonia contrast, expect missed REM"; OOD > 10% → "recording unlike
training data". Surface in UI/CLI/API as a reliability banner, not as per-epoch blanking.

**Serving design concern (RESOLVED 2026-11-13):** the backend used to map per-epoch OOD → "Unscorable".
At q=0.999 that would blank 15–30% of epochs on recordings like ZZ05/ZZ27/ZZ30. Now implemented as
proposed: per-epoch `Unscored` only from signal QC (Layer A); OOD %, QC-fail % and EMG p05/p50 are
returned as a recording-level `reliability` block (`sleep_features/reliability.py`) and shown as a
banner in the UI / a warning block in the CLI.
