#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def best_keypoint_from_cluster(cluster: Dict[str, Any]) -> tuple[str, float]:
    """
    Pick the edge with the highest weight and return its kp (or empty string if no edges).
    """
    edges = cluster.get("edges", [])
    if not edges:
        return "", 0.0
    best = max(edges, key=lambda e: e.get("weight", 0.0))
    return best.get("kp", ""), float(best.get("weight", 0.0))


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="(Part 2) Read clusters.jsonl and output keypoints.jsonl"
    )
    p.add_argument("--clusters_in", type=str, required=True,
                   help="Path to clusters.jsonl created by 01_make_clusters.py")
    p.add_argument("--keypoints_out", type=str, default="./output/keypoints.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    clusters_path = Path(args.clusters_in)
    out_path = Path(args.keypoints_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with clusters_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            cluster = json.loads(line)
            kp, w = best_keypoint_from_cluster(cluster)
            fout.write(json.dumps({
                "topic": cluster["topic"],
                "cluster_id": cluster["cluster_id"],
                "keypoint": kp,
                "best_edge_weight": w
            }, ensure_ascii=False) + "\n")

    print(f"Keypoints written to {args.keypoints_out}")


if __name__ == "__main__":
    main()