#!/usr/bin/env python3
"""Utilities to merge numbered topics and keep diverse key points.

The CSVs produced by the Khosravani key point generation pipeline sometimes
contain one logical topic that was split into several numbered variants such as
"Schooluniform - 1", "Schooluniform - 2", and so on. This helper script groups
those variants by their base topic name, compares their key points with the
Khosravani sentence encoder, and keeps only semantically distinct ones per
(merged) topic.

Example
-------
python reduce_keypoints_from_multiple_topics.py \
    --input ./keypoints_split.csv \
    --output ./keypoints_reduced.csv \
    --similarity-threshold 0.78 \
    --include-source-topics

This keeps one representative key point per similarity cluster while recording
which original topic variants it came from (with ``--include-source-topics``).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


_TOPIC_SUFFIX_RE = re.compile(r"\s*-\s*\d+\s*[\.\)]?\s*$")


@dataclass
class ReducedKeyPoint:
    """Container for a reduced key point and metadata."""

    record: Dict[str, object]
    embedding: np.ndarray
    source_topics: Set[str]


def _default_embedder_path() -> Path:
    """Return the default location of the Khosravani sentence encoder."""

    here = Path(__file__).resolve().parent
    model_path = here.parent / "khosravani2024" / "models" / "V1"
    return model_path


def _base_topic(topic: str) -> str:
    """Strip trailing " - <number>" patterns from topic names."""

    if topic is None:
        return ""
    return _TOPIC_SUFFIX_RE.sub("", str(topic)).strip()


def _encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Compute L2-normalised embeddings for the given texts."""

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _reduce_group(
    base_topic: str,
    group_df: pd.DataFrame,
    args: argparse.Namespace,
    model: SentenceTransformer,
) -> List[Dict[str, object]]:
    """Reduce key points for a single base topic."""

    topic_col = args.topic_col
    keypoint_col = args.keypoint_col

    working_df = group_df.copy()
    working_df[keypoint_col] = working_df[keypoint_col].astype(str)
    working_df[topic_col] = working_df[topic_col].astype(str)
    working_df = working_df.assign(__kp_norm=working_df[keypoint_col].str.strip())
    working_df = working_df[working_df["__kp_norm"] != ""]

    if working_df.empty:
        return []

    if args.sort_by:
        ascending = bool(args.sort_ascending)
        if args.sort_by not in working_df.columns:
            raise ValueError(
                f"Column '{args.sort_by}' not found but was passed to --sort-by."
            )
        working_df = working_df.sort_values(
            by=args.sort_by,
            ascending=ascending,
            kind="mergesort",  # stable to keep deterministic tie-breaking
        )
    # reset index so we can map embeddings back easily
    working_df = working_df.reset_index(drop=True)

    texts = working_df[keypoint_col].tolist()
    embeddings = _encode_texts(model, texts, args.batch_size)

    reduced: List[ReducedKeyPoint] = []

    for idx, embedding in enumerate(embeddings):
        row = working_df.iloc[idx].to_dict()
        original_topic = row[topic_col]

        if not reduced:
            reduced.append(
                ReducedKeyPoint(
                    record=row,
                    embedding=embedding,
                    source_topics={original_topic},
                )
            )
        else:
            stacked = np.vstack([r.embedding for r in reduced])
            sims = stacked @ embedding
            best_match_idx = int(np.argmax(sims))
            best_sim = float(sims[best_match_idx])
            if best_sim >= args.similarity_threshold:
                reduced[best_match_idx].source_topics.add(original_topic)
                continue
            reduced.append(
                ReducedKeyPoint(
                    record=row,
                    embedding=embedding,
                    source_topics={original_topic},
                )
            )
        if args.max_per_group is not None and len(reduced) >= args.max_per_group:
            break

    outputs: List[Dict[str, object]] = []
    for entry in reduced:
        data = dict(entry.record)
        data.pop("__kp_norm", None)
        data.pop("__base_topic", None)
        data[topic_col] = base_topic
        if args.include_source_topics:
            data["source_topics"] = "; ".join(sorted(entry.source_topics))
            data["source_topic_count"] = len(entry.source_topics)
        outputs.append(data)

    return outputs


def _summarise(stats: List[tuple], verbose: bool, threshold: float) -> None:
    total_before = sum(before for _, before, _ in stats)
    total_after = sum(after for _, _, after in stats)
    num_topics = len(stats)
    print(
        f"[reduce] processed {total_before} key points "
        f"across {num_topics} merged topics -> {total_after} remain "
        f"(threshold={threshold:.2f})"
    )
    if verbose:
        for topic, before, after in stats:
            print(f"  - {topic}: {before} -> {after}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge numbered topic variants, keep diverse key points using the "
            "Khosravani sentence encoder."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the reduced CSV",
    )
    parser.add_argument(
        "--topic-col",
        default="topic",
        help="Column name for topics in the input CSV",
    )
    parser.add_argument(
        "--keypoint-col",
        default="key_point",
        help="Column name containing the key point text",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Optional column used to prioritise rows within a topic before reduction",
    )
    parser.add_argument(
        "--sort-ascending",
        action="store_true",
        help="Sort ascending when --sort-by is supplied (default: descending)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.58,
        help=(
            "Cosine similarity threshold; >= threshold is treated as duplicate. "
            "Lower values keep fewer key points."
        ),
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Optional hard cap on the number of key points to keep per merged topic",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size"
    )
    parser.add_argument(
        "--embedder-path",
        default="../khosravani2024/models/V1",
        help="Path to the SentenceTransformer model (defaults to Khosravani V1)",
    )
    parser.add_argument(
        "--include-source-topics",
        action="store_true",
        help="Add a column listing the original topic variants that mapped to each key point",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-topic before/after counts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    embedder_path = (
        Path(args.embedder_path) if args.embedder_path else _default_embedder_path()
    )
    if not embedder_path.exists():
        raise FileNotFoundError(
            f"SentenceTransformer model not found: {embedder_path}. "
            "Use --embedder-path to point to the Khosravani encoder."
        )

    df = pd.read_csv(input_path)
    if args.topic_col not in df.columns:
        raise KeyError(f"Column '{args.topic_col}' is missing from {input_path}")
    if args.keypoint_col not in df.columns:
        raise KeyError(f"Column '{args.keypoint_col}' is missing from {input_path}")

    df = df.copy()
    df[args.topic_col] = df[args.topic_col].astype(str)
    df["__base_topic"] = df[args.topic_col].apply(_base_topic)

    model = SentenceTransformer(str(embedder_path))

    stats = []
    reduced_records: List[Dict[str, object]] = []
    for base_topic, group_df in df.groupby("__base_topic", sort=True):
        before = len(group_df)
        reduced = _reduce_group(base_topic, group_df, args, model)
        reduced_records.extend(reduced)
        after = len(reduced)
        stats.append((base_topic, before, after))

    output_df = pd.DataFrame(reduced_records)
    if output_df.empty:
        print("[reduce] no key points survived the filtering; writing empty CSV.")
    else:
        # Deterministic order by topic then optionally sort key points by retained order.
        order_cols = [args.topic_col]
        if args.sort_by and args.sort_by in output_df.columns:
            order_cols.append(args.sort_by)
        output_df = output_df.sort_values(by=order_cols, ascending=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    _summarise(stats, args.verbose, args.similarity_threshold)
    print(f"[reduce] wrote {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
