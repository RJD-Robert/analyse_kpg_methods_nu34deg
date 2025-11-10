#!/usr/bin/env python3
"""
cluster_args.py

Reads a CSV with columns: topic, argument
→ clusters arguments per topic using SentenceTransformers + AgglomerativeClustering
→ writes clusters as JSONL (one line per topic)

Example:
    python cluster_args.py \
        --arguments-file ../data/shared_task_data/arguments_combined.csv \
        --embedder ./models/V1 \
        --num-keypoints 10 \
        --out clusters.jsonl
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def build_topic_args_dict(path_to_arguments_csv: str):
    """Return dict[topic]['ALL'] -> list[str] arguments."""
    df = pd.read_csv(path_to_arguments_csv)
    topic_args = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        topic_args[row["topic"]]["ALL"].append(row["argument"])
    return topic_args


def cluster_topics(input_args_kp_by_topic, embedder, num_keypoints=None, distance_threshold=None):
    """
    Perform Agglomerative Clustering per topic.
    If num_keypoints is given, a fixed number of clusters is used for every topic.
    Otherwise, distance_threshold is used.
    Returns: dict[topic][cluster_id] -> list[str] arguments
    """
    assert (num_keypoints is not None) ^ (distance_threshold is not None), \
        "Provide exactly one of --num-keypoints or --distance-threshold."

    cluster_by_topic = {}
    topics = list(input_args_kp_by_topic.keys())

    for topic in tqdm(topics, desc="Clustering topics"):
        arguments = list(set().union(*input_args_kp_by_topic[topic].values()))
        if not arguments:
            cluster_by_topic[topic] = {}
            continue

        # Embed & normalize
        corpus_embeddings = embedder.encode(arguments)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        if num_keypoints is not None:
            clustering_model = AgglomerativeClustering(n_clusters=num_keypoints)
        else:
            clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)

        clustering_model.fit(corpus_embeddings)
        labels = clustering_model.labels_

        topic_clusters = defaultdict(list)
        for sent_id, cl_id in enumerate(labels):
            topic_clusters[int(cl_id)].append(arguments[sent_id])

        cluster_by_topic[topic] = topic_clusters

    return cluster_by_topic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arguments-file", required=True, help="CSV with columns: topic, argument")
    parser.add_argument("--embedder", required=True, help="SentenceTransformer model path or name")
    parser.add_argument("--num-keypoints", type=int, default=None,
                        help="Fixed number of clusters per topic")
    parser.add_argument("--distance-threshold", type=float, default=None,
                        help="Alternatively cluster with a distance threshold (no fixed K)")
    parser.add_argument("--out", required=True, help="Output JSONL file with clusters")
    args = parser.parse_args()

    t0 = time.time()
    topic_args_only = build_topic_args_dict(args.arguments_file)
    embedder = SentenceTransformer(args.embedder)

    clusters = cluster_topics(
        topic_args_only,
        embedder,
        num_keypoints=args.num_keypoints,
        distance_threshold=args.distance_threshold
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for topic, topic_clusters in clusters.items():
            # Ensure cluster IDs are JSON-serialisable keys
            rec = {
                "topic": topic,
                "clusters": {str(cid): sents for cid, sents in topic_clusters.items()}
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(clusters)} topics to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()