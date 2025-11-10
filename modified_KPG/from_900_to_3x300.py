#!/usr/bin/env python3
"""
Split each topic's arguments into up to three sub-topics with at most 300 arguments each.

Input CSV columns (required):
  - topic, stance, argument, quality_score

Rules:
  - For every original topic, randomly distribute its arguments into up to 3 parts.
  - Each part may contain at most 300 arguments.
  - If a topic has fewer than 900 arguments, split into as many parts as needed (up to 3),
    never exceeding 300 rows per part.
  - If a topic has more than 900 arguments, this script will raise an error unless
    `--allow-more` is provided, in which case it will still create only 3 parts of 300 each
    (first 900 after shuffling) and will drop the remainder with a warning.
  - The new topic names will be: "<topic> - 1", "<topic> - 2", "<topic> - 3" (as needed).

Usage:
  python make_900_to_3x300.py --input INPUT.csv --output OUTPUT.csv [--seed 42] [--allow-more]
"""

from __future__ import annotations
import argparse
import math
import sys
import warnings
from typing import List

import pandas as pd


def split_topic(
    df_topic: pd.DataFrame, seed: int | None, allow_more: bool
) -> List[pd.DataFrame]:
    n = len(df_topic)
    if n == 0:
        return []

    if n > 900 and not allow_more:
        raise ValueError(
            f"Topic '{df_topic['topic'].iloc[0]}' has {n} arguments (>900). "
            "Use --allow-more to proceed (it will keep only 900 after shuffle)."
        )

    # Shuffle rows for random distribution
    if seed is None:
        shuffled = df_topic.sample(frac=1, random_state=None).reset_index(drop=True)
    else:
        shuffled = df_topic.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Determine number of parts needed, capped at 3
    num_parts = min(3, math.ceil(min(n, 900) / 300))

    parts: List[pd.DataFrame] = []
    for i in range(num_parts):
        start = i * 300
        end = min((i + 1) * 300, len(shuffled))
        if start >= end:
            break
        part = shuffled.iloc[start:end].copy()
        original_topic = part["topic"].iloc[0]
        part["topic"] = f"{original_topic} - {i + 1}"
        parts.append(part)

    if n > 900 and allow_more:
        dropped = n - 900
        if dropped > 0:
            warnings.warn(
                f"Topic '{df_topic['topic'].iloc[0]}' had {n} arguments; dropping {dropped} to keep 3×300.",
                RuntimeWarning,
            )

    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Split topics into up to 3×300 random parts."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--allow-more",
        action="store_true",
        help="If a topic has >900 arguments, keep only 900 (3×300) after shuffle and drop the rest",
    )

    args = parser.parse_args()

    # Read CSV
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"topic", "stance", "argument", "quality_score"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Input CSV missing required columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # Group by topic and split
    out_frames: List[pd.DataFrame] = []
    for topic, df_topic in df.groupby("topic", sort=False):
        parts = split_topic(df_topic, seed=args.seed, allow_more=args.allow_more)
        out_frames.extend(parts)

    if not out_frames:
        print("No data produced.", file=sys.stderr)
        sys.exit(1)

    df_out = pd.concat(out_frames, ignore_index=True)

    # Keep original column order if possible
    cols = [
        c
        for c in ["topic", "stance", "argument", "quality_score"]
        if c in df_out.columns
    ]
    other_cols = [c for c in df_out.columns if c not in cols]
    df_out = df_out[cols + other_cols]

    # Write CSV
    try:
        df_out.to_csv(args.output, index=False)
    except Exception as e:
        print(f"Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        "Done. Wrote split CSV to",
        args.output,
        f"with {len(df_out)} rows across up to 3 parts per topic.",
    )


if __name__ == "__main__":
    main()
