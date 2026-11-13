"""
Per-animal REM diagnosis — why does the model miss REM on a given animal?

For every animal (leave-one-animal-out predictions, no filter, no REM weighting)
this reports four complementary views, so the failure mode can be classified:

  A. Error anatomy      where missed REM goes (Wake vs NREM), where false REM comes
                        from, how confident the model was on the misses (is it a
                        threshold / operating-point problem?), and whether whole
                        REM bouts are missed or just their edges.
  B. Separability       within-animal ROC-AUC of key features for REM vs rest.
                        High AUC + low recall  => boundary misplaced (fixable with
                        data / weighting).  Low AUC => weak signal (not fixable by
                        more animals).
  C. Distribution shift z-score of the animal's REM feature medians against the
                        other animals' REM medians (is it an unusual animal?).
  D. Raw signal         from the WAV: EEG theta-peak prominence and theta/delta in
                        REM, EMG atonia depth, high-frequency noise, 60 Hz line
                        contamination, QC-flagged fraction, raw gain.

Optionally (E) a label-quality check for animals that ship two disagreeing score
files: on disputed epochs, which file does the model side with?

Usage:
    python -m model.experiments.diagnose_animals --v2 feature_store/<v2>.csv \
        --data "training_data/Filtered Scores 2" [--skip-raw] [--focus ZZ01 ZZ05 ZZ27]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from sleep_features.config import FEATURE_SETS, UNSCORED_LABEL
from sleep_features.discovery import discover_recordings
from sleep_features.io import load_wav
from sleep_features.qc import compute_qc_flags
from sleep_features.scores import align_scores, load_scores
from model.experiments.rem_phase23_experiment import augment_features
from model.experiments.rem_recall_experiment import GROUP_COL, REM, TARGET_COL, TIME_COL, _fit

KEY_FEATS = ["theta_over_delta", "theta_rel", "theta_dominance", "delta_rel",
             "EMG_std", "emg_atonia", "sigma_rel", "spectral_entropy"]


# ----------------------------------------------------------------- LOO w/ proba
def loo_predictions(df, cols):
    enc = LabelEncoder().fit(df[TARGET_COL])
    classes = enc.classes_
    rem_idx = list(classes).index(REM)
    out = []
    for a in df[GROUP_COL].unique():
        tr = df.loc[df[GROUP_COL] != a]
        te = df.loc[df[GROUP_COL] == a].sort_values(TIME_COL)
        m = _fit(tr[cols], enc.transform(tr[TARGET_COL]), (tr[TARGET_COL] == REM).to_numpy(), 1)
        proba = m.predict_proba(te[cols])
        out.append(pd.DataFrame({
            GROUP_COL: a, TIME_COL: te[TIME_COL].to_numpy(),
            "y": te[TARGET_COL].to_numpy(), "pred": classes[proba.argmax(1)],
            "p_rem": proba[:, rem_idx],
        }))
    return pd.concat(out, ignore_index=True)


def _bouts(mask):
    """Start/end indices of runs of True in a boolean array."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2] - 1))


def section_a(sub):
    y, p, prem = sub["y"].to_numpy(), sub["pred"].to_numpy(), sub["p_rem"].to_numpy()
    is_rem = y == REM
    n_rem = is_rem.sum()
    missed = is_rem & (p != REM)
    false_rem = ~is_rem & (p == REM)
    r = {
        "REM epochs": int(n_rem), "REM %": 100 * n_rem / len(y),
        "recall": 1 - missed.sum() / max(n_rem, 1),
        "precision": (is_rem & (p == REM)).sum() / max((p == REM).sum(), 1),
        "missed->Wake": int((missed & (p == "Wake")).sum()),
        "missed->NREM": int((missed & (p == "Non REM")).sum()),
        "falseREM<-Wake": int((false_rem & (y == "Wake")).sum()),
        "falseREM<-NREM": int((false_rem & (y == "Non REM")).sum()),
        "missed: median p(REM)": float(np.median(prem[missed])) if missed.any() else np.nan,
        "missed: % p(REM)>=0.2": 100 * (prem[missed] >= 0.2).mean() if missed.any() else np.nan,
        "missed: % p(REM)>=0.35": 100 * (prem[missed] >= 0.35).mean() if missed.any() else np.nan,
    }
    bouts = _bouts(is_rem)
    if bouts:
        lens = np.array([e - s + 1 for s, e in bouts])
        wholly_missed = sum(1 for s, e in bouts if (p[s:e + 1] != REM).all())
        edge = np.zeros(len(y), bool)
        for s, e in bouts:
            edge[s] = True; edge[e] = True
        r.update({
            "REM bouts": len(bouts), "median bout len (epochs)": float(np.median(lens)),
            "% bouts wholly missed": 100 * wholly_missed / len(bouts),
            "% singleton bouts": 100 * (lens == 1).mean(),
            "% missed epochs at bout edge": 100 * edge[missed].mean() if missed.any() else np.nan,
        })
    # label plausibility
    prev = np.r_[None, y[:-1]]
    rem_start = is_rem & (prev != REM)
    r["% REM bouts entered from Wake"] = 100 * ((prev[rem_start] == "Wake").mean()) if rem_start.any() else np.nan
    return r


def section_b(feat_sub):
    y = (feat_sub[TARGET_COL] == REM).to_numpy()
    out = {}
    for f in KEY_FEATS:
        if f in feat_sub:
            x = feat_sub[f].to_numpy()
            ok = np.isfinite(x)
            auc = roc_auc_score(y[ok], x[ok])
            out[f] = max(auc, 1 - auc)  # direction-agnostic
    return out


def section_c(feats):
    """z of each animal's REM-median vs the other animals' REM medians."""
    med = feats.loc[feats[TARGET_COL] == REM].groupby(GROUP_COL)[KEY_FEATS].median()
    z = pd.DataFrame(index=med.index, columns=med.columns, dtype=float)
    for a in med.index:
        others = med.drop(index=a)
        z.loc[a] = (med.loc[a] - others.mean()) / (others.std(ddof=1) + 1e-9)
    return med, z


# ----------------------------------------------------------------- raw signal
def section_d(rec, labels, window=10):
    df, sr = load_wav(rec.wav_path)
    n = len(df) // (window * sr)
    eeg = df["eeg"].to_numpy()[: n * window * sr].reshape(n, window * sr)
    emg = df["emg"].to_numpy()[: n * window * sr].reshape(n, window * sr)
    dt = 1 / sr
    X = rfft(eeg - eeg.mean(1, keepdims=True), axis=1)
    psd = (2 * dt**2 / window * (X * X.conj())).real  # 0.1 Hz bins
    f = np.arange(psd.shape[1]) / window

    def band(lo, hi):
        return psd[:, (f >= lo) & (f < hi)]
    delta = band(0.5, 4).sum(1); theta = band(5.5, 8.5).sum(1)
    total = band(0.5, 45).sum(1); hf = band(20, 45).sum(1)
    theta_peak = band(5.5, 8.5).max(1)
    flank = np.concatenate([band(3.5, 5.0), band(9.0, 10.5)], axis=1).mean(1)
    line = band(59.5, 60.5).mean(1)
    line_flank = np.concatenate([band(55, 58), band(62, 65)], axis=1).mean(1)
    eeg_std = eeg.std(1); emg_std = emg.std(1)

    lab = np.asarray(labels[:n])
    st = {s: lab == s for s in ("Wake", "Non REM", REM)}
    med = lambda x, m: float(np.median(x[m])) if m.any() else np.nan

    qc = compute_qc_flags(df, window=window, samplerate=sr)
    r = {
        "sr": sr, "hours": n * window / 3600,
        "QC flagged %": 100 * qc["qc_flag"].mean(),
        "EEG std Wake (raw units)": med(eeg_std, st["Wake"]),
        "EMG std Wake (raw units)": med(emg_std, st["Wake"]),
        "REM theta-peak prominence": med(theta_peak / (flank + 1e-12), st[REM]),
        "NREM theta-peak prominence": med(theta_peak / (flank + 1e-12), st["Non REM"]),
        "REM theta/delta (raw)": med(theta / (delta + 1e-12), st[REM]),
        "NREM theta/delta (raw)": med(theta / (delta + 1e-12), st["Non REM"]),
        "Wake theta/delta (raw)": med(theta / (delta + 1e-12), st["Wake"]),
        "EMG atonia: REM/Wake": med(emg_std, st[REM]) / med(emg_std, st["Wake"]),
        "EMG atonia: REM/NREM": med(emg_std, st[REM]) / med(emg_std, st["Non REM"]),
        "HF-noise 20-45/0.5-45 (REM)": med(hf / (total + 1e-12), st[REM]),
        "60 Hz line ratio": float(np.median(line / (line_flank + 1e-12))),
    }
    del df, eeg, emg, X, psd
    return r


# ----------------------------------------------------------------- label check
def section_e(rec, preds_sub, n_epochs):
    files = [Path(p) for p in glob.glob(os.path.join(rec.wav_path.parent, "*scores*.csv"))]
    others = [p for p in files if p.resolve() != rec.scores_path.resolve()]
    if not others:
        return None
    alt = align_scores(load_scores(others[0]), n_epochs)
    used = align_scores(load_scores(rec.scores_path), n_epochs)
    pred = pd.Series(preds_sub["pred"].to_numpy(), index=preds_sub[TIME_COL].to_numpy())
    both = pd.DataFrame({"used": used, "alt": alt}).join(pred.rename("pred"), how="inner")
    both = both[(both.used != UNSCORED_LABEL) & (both.alt != UNSCORED_LABEL)]
    disp = both[both.used != both.alt]
    r = {
        "alt file": others[0].name, "n compared": len(both), "n disputed": len(disp),
        "model agrees w/ USED file on disputed %": 100 * (disp.pred == disp.used).mean(),
        "model agrees w/ ALT file on disputed %": 100 * (disp.pred == disp.alt).mean(),
        "model agrees w/ neither on disputed %": 100 * ((disp.pred != disp.used) & (disp.pred != disp.alt)).mean(),
    }
    # The disputed epochs that are REM in the used file: does the model call REM?
    d_rem = disp[disp.used == REM]
    r["disputed REM(used) epochs"] = len(d_rem)
    r["  model says REM %"] = 100 * (d_rem.pred == REM).mean() if len(d_rem) else np.nan
    r["  alt file says"] = d_rem.alt.value_counts().to_dict()
    return r


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 1000 else f"{v:.3g}"
    return str(v)


def print_table(title, rows: dict, cols):
    print(f"\n{title}")
    keys = list(next(iter(rows.values())).keys())
    w = max(len(k) for k in keys) + 2
    print(" " * w + "".join(f"{c:>12}" for c in cols))
    for k in keys:
        print(f"{k:<{w}}" + "".join(f"{_fmt(rows[c].get(k, '')):>12}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--focus", nargs="*", default=None, help="animal IDs to print raw/label detail for")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.v2)
    df = df.loc[df[TARGET_COL] != UNSCORED_LABEL].copy()
    df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    v2 = [c for c in FEATURE_SETS["v2"] if c in df.columns]
    df, extra = augment_features(df)
    cols = v2 + extra
    animals = list(df[GROUP_COL].unique())
    short = {a: a.split("_")[0] for a in animals}

    print(f"{len(df)} scored epochs, {len(animals)} animals, {len(cols)} features (v2 + REM extras)")
    preds = loo_predictions(df, cols)

    A = {short[a]: section_a(preds[preds[GROUP_COL] == a]) for a in animals}
    print_table("A. ERROR ANATOMY (LOO, no filter, no REM weighting)", A, list(A))

    B = {short[a]: section_b(df[df[GROUP_COL] == a]) for a in animals}
    print_table("B. WITHIN-ANIMAL SEPARABILITY — ROC-AUC, REM vs rest (1.0 = perfectly separable)", B, list(B))

    med, z = section_c(df)
    med.index = [short[a] for a in med.index]; z.index = med.index
    print("\nC. DISTRIBUTION SHIFT — REM-epoch feature medians (normalized units)")
    print(med.round(2).T.to_string())
    print("\n   z-score of each animal's REM median vs the OTHER animals (|z|>2 = unusual animal)")
    print(z.round(1).T.to_string())

    recs = {r.id: r for r in discover_recordings(args.data)}
    focus = args.focus or list(short.values())

    if not args.skip_raw:
        D = {}
        for a in animals:
            s = short[a]
            if s not in focus or s not in recs:
                continue
            rec = recs[s]
            n_epochs = int(df.loc[df[GROUP_COL] == a, TIME_COL].max()) + 1
            labels = align_scores(load_scores(rec.scores_path), 10**6).to_numpy()
            print(f"   raw: {rec.wav_path.name} ...", flush=True)
            D[s] = section_d(rec, labels)
        if D:
            print_table("D. RAW SIGNAL (from WAV; medians over epochs of the given state)", D, list(D))

    print("\nE. LABEL-QUALITY CHECK (animals with a second, disagreeing score file)")
    for a in animals:
        s = short[a]
        if s not in focus or s not in recs:
            continue
        n_epochs = int(df.loc[df[GROUP_COL] == a, TIME_COL].max()) + 1
        r = section_e(recs[s], preds[preds[GROUP_COL] == a], n_epochs)
        if r:
            print(f"  {s}:")
            for k, v in r.items():
                print(f"    {k:<42} {_fmt(v) if not isinstance(v, dict) else v}")


if __name__ == "__main__":
    main()
