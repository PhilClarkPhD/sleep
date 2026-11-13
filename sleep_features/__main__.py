"""
Command-line entry point for the feature-store operator.

Usage:
    python -m sleep_features build --input training_data/ --out feature_store/
    python -m sleep_features build --input training_data/ --out feature_store/ \
        --normalization robust_mad --limit 2
"""

import argparse
import json
import logging
import sys


def _build(args) -> int:
    from .build import build_feature_store

    manifest = build_feature_store(
        input_dir=args.input,
        out_dir=args.out,
        normalization=args.normalization,
        emg_event_k=args.emg_event_k,
        feature_set=args.feature_set,
        limit=args.limit,
    )
    totals = manifest["totals"]
    print(
        f"\nBuilt {manifest['_csv_path']}\n"
        f"  recordings: {totals['n_recordings']}\n"
        f"  rows:       {totals['n_rows']}\n"
        f"  labels:     {json.dumps(totals['label_distribution'])}\n"
        f"  manifest:   {manifest['_manifest_path']}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sleep_features")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build a versioned feature store from raw recordings")
    b.add_argument("--input", required=True, help="Directory of raw recordings (walked recursively)")
    b.add_argument("--out", required=True, help="Output directory for the feature CSV + manifest")
    b.add_argument(
        "--normalization",
        default="robust_mad",
        choices=["robust_mad", "baseline_epoch", "none"],
        help="Per-recording normalization strategy (default: robust_mad)",
    )
    b.add_argument("--emg-event-k", type=float, default=2.0, help="EMG event-threshold multiplier")
    b.add_argument(
        "--feature-set",
        default="v1",
        choices=["v1", "v2"],
        help="v1 (current 12 features) or v2 (adds sigma/beta/gamma, entropy, Hjorth, temporal context)",
    )
    b.add_argument("--limit", type=int, default=None, help="Only build the first N recordings")
    b.set_defaults(func=_build)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
