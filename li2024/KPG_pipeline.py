#!/usr/bin/env python3

from __future__ import annotations

# ============================================================================
# English overview
# ----------------------------------------------------------------------------
# This module implements a two-phase pipeline for extracting concise keypoints
# (KPs) from argument pairs about the same topic.
#
# PHASE 1 (Data prep & model inference)
#   • Build all unordered argument pairs per topic.
#   • Run a seq2seq model (e.g., Flan‑T5) that outputs a label-like first token
#     (ideally "Yes"/"No") plus a free‑form explanation. We treat the first
#     token as an affinity signal between the paired arguments and use its
#     probability as an edge weight.
#   • Results are cached to avoid recomputation.
#
# PHASE 2 (Graph clustering & keypoint selection)
#   • Build a topic graph whose nodes are unique arguments; edges exist only for
#     pairs predicted as "Yes" and carry the weighted explanation text.
#   • Split the graph into subgraphs using sentence‑embedding clustering.
#   • Softly refine partitions by moving nodes if the average intra‑subgraph
#     edge weight improves.
#   • Pick the strongest edge in each subgraph as a candidate keypoint; fill up
#     to N keypoints with the highest‑weighted remaining edges per topic.
#
# Design notes
#   • Logging/printing is intentionally concise and happens at stage boundaries
#     only (not inside tight loops) to keep the console readable.
#   • Extensive inline comments explain non‑obvious steps for new readers.
# ============================================================================

import json
import math
import random
import re
import hashlib
from pathlib import Path
import copy
from typing import Tuple, List, Dict, Optional

import numpy as np
import networkx as nx
import pandas as pd
import torch
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, PreTrainedTokenizer

print("#" * 35)
print("USING CUDA" if torch.cuda.is_available() else "USING CPU")
print("#" * 35)


# -----------------------------
# --------- PHASE 1 -----------
#  Argument-Paar-Generierung &
#  Modell-Inferenz
# -----------------------------


def normalize_stance(raw: str) -> str:
    """Normalize a stance label to one of {"positive", "negative"} when possible.

    Accepted variants for positive: {"1", "+", "positive", "positiv", "pos"}.
    Accepted variants for negative: {"-1", "-", "negative", "negativ", "neg"}.

    Any other input is returned unchanged (lowercased/stripped), so the caller can
    choose how to handle unexpected labels. This keeps the function robust to
    noisy, multilingual inputs (EN/DE)."""
    val = str(raw).strip().lower()
    if val in {"1", "+", "positive", "positiv", "pos"}:
        return "positive"
    if val in {"-1", "-", "negative", "negativ", "neg"}:
        return "negative"
    return val


def build_samples(df: pd.DataFrame) -> list[dict]:
    """Create unordered argument pairs (i < j) for each topic.

    For each topic group, this function generates all unique, order‑invariant pairs
    of arguments. The seq2seq input format is:
        "<topic> | <stance1>. <argument1> | <stance2>. <argument2>"

    Returns a list of dicts with fields:
    - 'id': a stable identifier equal to 'input' (also used as cache key)
    - 'input': the prompt string fed into the seq2seq model
    """
    samples = []
    # Group by topic so that pairs are only formed within the same discussion.
    for topic, group in df.groupby("topic", sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                stance1 = normalize_stance(group.at[i, "stance"])
                stance2 = normalize_stance(group.at[j, "stance"])
                arg1 = str(group.at[i, "argument"]).strip()
                arg2 = str(group.at[j, "argument"]).strip()
                # Construct a compact, reproducible prompt for the generator.
                input_str = f"{topic.strip()} | {stance1}. {arg1} | {stance2}. {arg2}"
                samples.append({"id": input_str, "input": input_str})
    return samples


# -----------------------------
#  Helper functions for cache path resolution
# -----------------------------


def _slugify(text: str) -> str:
    """Conservative slug for directory/file names (ASCII whitelist)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-").lower()


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA1 fingerprint over the relevant columns of the dataset.

    Uses a row-wise update with an explicit separator to avoid collisions
    like ("ab", "c") vs ("a", "bc"). Only columns that influence the
    seq2seq input are considered: topic, stance, argument (if present).
    Returns a short hex digest to keep paths compact.
    """
    cols = [c for c in ["topic", "stance", "argument"] if c in df.columns]
    h = hashlib.sha1()
    for row in df[cols].itertuples(index=False, name=None):
        parts = ["" if x is None else str(x) for x in row]
        h.update(("\x1f".join(parts) + "\n").encode("utf-8"))
    return h.hexdigest()[:16]


def resolve_cache_path(args, df: pd.DataFrame) -> Path:
    """Build a cache file path that scopes by dataset + model + gen-params.

    Layout:
      <cache_dir>/<namespace?>/<datasetName>-<datasetSHA>/<model+genSig>/generate_cache.jsonl
    where genSig includes the input and output token lengths.
    """
    base = Path(getattr(args, "cache_dir", "./output/cache"))
    namespace = getattr(args, "cache_namespace", "")
    ns = _slugify(namespace) if namespace else None

    dataset_name = _slugify(Path(args.input_csv).stem)
    dataset_fp = compute_dataset_fingerprint(df)
    model_id = _slugify(
        Path(args.seq2seq_model_path).name or str(args.seq2seq_model_path)
    )
    gen_sig = f"m={model_id}_in={args.max_input_length}_out={args.max_new_tokens}"

    cache_dir = base
    if ns:
        cache_dir = cache_dir / ns
    cache_dir = cache_dir / f"{dataset_name}-{dataset_fp}" / gen_sig
    return cache_dir / "generate_cache.jsonl"


class DataLoaderKPA:
    """Minimal batch loader for seq2seq generation.

    It tokenizes a list of query strings with padding/truncation for efficient
    batched generation. We keep it lightweight and framework‑agnostic so that it
    works with any Hugging Face-compatible tokenizer/model.
    """

    def __init__(
        self,
        ids: list[str],
        queries: list[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int,
    ):
        self.ids = ids
        self.queries = queries
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __len__(self) -> int:
        return (len(self.ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.ids), self.batch_size):
            end = start + self.batch_size
            batch_ids = self.ids[start:end]
            batch_queries = self.queries[start:end]
            tokenized = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            yield batch_ids, tokenized


def load_seq2seq_model(model_path: str, device: torch.device):
    """Load a Hugging Face seq2seq model + tokenizer and place the model on device.

    If CUDA is available we cast the model to half precision (fp16) to reduce
    memory usage and speed up inference; on CPU the model remains in full precision.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    if device.type == "cuda":
        model = model.half()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    ids: list[str],
    queries: list[str],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_input_length: int,
    cache_path: Path,
) -> dict[str, dict]:
    """
    Generate model outputs in batches with on-disk caching and basic confidence.

    Parameters
    ----------
    model, tokenizer : HF model + tokenizer used for generation
    ids, queries     : parallel lists; each id is a stable cache key for the query
    device           : torch.device used for compute
    batch_size       : generation batch size
    max_new_tokens   : maximum tokens generated per sequence
    max_input_length : tokenizer truncation length
    cache_path       : JSONL file storing {id, data} entries to skip recomputation

    Returns
    -------
    Dict[str, dict] mapping each id -> {
      'output'           : decoded text
      'confidence_score' : P(Yes) from the first-token softmax
      'max_token'        : argmax token string for the first position
      'raw_scores'       : raw first-position scores for [Yes, No]
    }

    Notes
    -----
    We **do not** print per-batch or per-item information to keep the console clean.
    A high-level summary is printed by the caller instead.
    """
    outputs: dict[str, dict] = {}
    processed_ids: set[str] = set()
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry["id"])
                outputs[entry["id"]] = entry["data"]
        print(f"[CACHE] Loaded entries: {len(processed_ids)}")
    print(f"[GENERATE] {len(ids) - len(processed_ids)} items remaining")

    remaining = [(i, q) for i, q in zip(ids, queries) if i not in processed_ids]
    if not remaining:
        print("[CACHE] All items already processed.")
        return outputs

    rem_ids, rem_queries = zip(*remaining)
    dataloader = DataLoaderKPA(
        list(rem_ids), list(rem_queries), tokenizer, batch_size, max_input_length
    )
    model.eval()

    # Identify token ids for the label-like first token so we can estimate
    # P(Yes) from the first decoder step (common in classification-by-generation).
    yes_token = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_token = tokenizer("No", add_special_tokens=False).input_ids[0]

    cache_file = cache_path.open("a", encoding="utf-8")

    for batch_ids, batch in tqdm(dataloader, desc="Generating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Decode full sequences once per batch; avoid any per-item prints here.
        decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        # First-token confidence
        first_scores = out.scores[0]
        yes_scores = first_scores[:, yes_token]
        no_scores = first_scores[:, no_token]
        raw_scores_tensor = torch.cat(
            (yes_scores.unsqueeze(-1), no_scores.unsqueeze(-1)), dim=-1
        )
        probs = torch.softmax(raw_scores_tensor, dim=-1)
        yes_probs = probs[:, 0].tolist()
        raw_scores = raw_scores_tensor.tolist()
        max_idxs = first_scores.argmax(dim=-1).tolist()
        max_tokens = tokenizer.convert_ids_to_tokens(max_idxs)

        for idx, sample_id in enumerate(batch_ids):
            data = {
                "output": decoded[idx],
                "confidence_score": yes_probs[idx],
                "max_token": max_tokens[idx],
                "raw_scores": raw_scores[idx],
            }
            outputs[sample_id] = data
            cache_file.write(
                json.dumps({"id": sample_id, "data": data}, ensure_ascii=False) + "\n"
            )

    cache_file.close()
    return outputs


def run_generation_phase(args, device: torch.device) -> list[dict]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Using device: {device}")

    # Load CSV
    df = pd.read_csv(args.input_csv)
    required = {"topic", "argument", "stance"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # ---- Dataset summary (concise; no per-row printing) ----
    topic_counts = df.groupby("topic", sort=False)["argument"].count()
    num_rows = len(df)
    num_topics = int(topic_counts.shape[0])
    min_args = int(topic_counts.min()) if num_topics else 0
    max_args = int(topic_counts.max()) if num_topics else 0
    avg_args = float(topic_counts.mean()) if num_topics else 0.0
    print(
        f"[SUMMARY][Data] rows={num_rows}, topics={num_topics}, arguments/topic: min={min_args}, avg={avg_args:.2f}, max={max_args}"
    )

    # Expected number of unordered pairs per topic (n choose 2) and totals
    pair_counts = (
        (topic_counts * (topic_counts - 1) // 2) if num_topics else pd.Series(dtype=int)
    )
    if num_topics:
        total_pairs_expected = int(pair_counts.sum())
        print(
            f"[SUMMARY][Pairs] expected_total={total_pairs_expected}, per-topic: min={int(pair_counts.min())}, avg={pair_counts.mean():.2f}, max={int(pair_counts.max())}"
        )

    # Samples
    # Sanity-check: the number of built samples should match the expected pairs.
    samples = build_samples(df)
    Path(args.input_json).write_text(json.dumps(samples, ensure_ascii=False, indent=4))
    print(f"{len(samples)} samples written to {args.input_json}")
    # Show the dataset fingerprint to make cache provenance explicit
    try:
        ds_fp = compute_dataset_fingerprint(df)
        print(f"[CACHE] dataset_id={_slugify(Path(args.input_csv).stem)}-{ds_fp}")
    except Exception:
        pass
    # Note: actual generation summary is printed after inference to avoid guessing.

    # Load model
    model, tokenizer = load_seq2seq_model(args.seq2seq_model_path, device)
    print(f"Seq2Seq model loaded from {args.seq2seq_model_path}")

    # Generate
    ids = [s["id"] for s in samples]
    queries = [s["input"] for s in samples]
    # Resolve per-dataset/per-model cache path (unless a custom --cache_path was explicitly given)
    if (
        hasattr(args, "cache_path")
        and args.cache_path
        and args.cache_path != "./output/generate_cache.jsonl"
    ):
        cache_path = Path(args.cache_path)
    else:
        cache_path = resolve_cache_path(args, df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE] Using cache file: {cache_path}")

    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        ids=ids,
        queries=queries,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        cache_path=cache_path,
    )

    # Final results list to feed into graph clustering
    results = []
    for s in samples:
        data = outputs[s["id"]]
        results.append(
            {
                "prompt": s["input"],
                "pred": data["output"],
                "confidence_score": data["confidence_score"],
                "max_token": data["max_token"],
                "raw_pred": data["output"],
                "raw_scores": data["raw_scores"],
            }
        )

    # ---- Generation summary (concise; aggregates only) ----
    total = len(results)
    yes_probs = [float(d.get("confidence_score", 0.0)) for d in results]
    # Heuristic: consider a prediction affirmative if it starts with "Yes" or
    # the first-token argmax equals the tokenizer token for "Yes".
    affirmative = [
        bool(re.match(r"^\s*Yes\b", str(d.get("pred", "")).strip()))
        or str(d.get("max_token", "")) in {"▁Yes", "Yes"}
        for d in results
    ]
    yes_count = sum(1 for v in affirmative if v)
    first_token_hist = {}
    for d in results:
        tok = str(d.get("max_token", ""))
        first_token_hist[tok] = first_token_hist.get(tok, 0) + 1

    if total > 0:
        conf_min = min(yes_probs)
        conf_avg = sum(yes_probs) / total
        conf_max = max(yes_probs)
        top_tokens = sorted(
            first_token_hist.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
        print(
            f"[SUMMARY][Gen] items={total}, 'Yes'={yes_count} ({yes_count / total:.1%}), "
            f"confidence: min={conf_min:.4f}, avg={conf_avg:.4f}, max={conf_max:.4f}"
        )
        print(f"[SUMMARY][Gen] most frequent first tokens: {top_tokens}")

    Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=4))
    print(f"Results written to {args.output_json}")

    return results


# -----------------------------
# --------- PHASE 2 -----------
#  Graph-Clustering & KP-Extr.
# -----------------------------


def normalize_topic_graph(text: str) -> str:
    """Ensure each topic string ends with exactly one trailing period.

    This normalization helps treat semantically identical topics as the same key in
    maps and file outputs, independent of minor punctuation differences.
    """
    return text.rstrip(" .") + "."


def find_core_sample(
    embeddings: np.ndarray, core_emb: Optional[np.ndarray] = None
) -> tuple[int, list[float]]:
    """Return the index of the most representative vector and all distances.

    The representative element is the one with the smallest Euclidean distance to
    `core_emb` (or to the mean if `core_emb` is None). Also returns the full list of
    per-sample distances so callers can perform secondary sorting.
    """
    if core_emb is None:
        core_emb = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - core_emb, axis=1)
    return int(distances.argmin()), distances.tolist()


def clustering_impl_kmeans(
    embs: np.ndarray, sentences: list[str], seed: int, n_clusters: int
) -> tuple[list[str], list[list[str]], None, dict]:
    """Cluster sentence embeddings with K-Means and pick a core item per cluster.

    Returns
    -------
    cores : List[str]
        One representative sentence per cluster (closest to the centroid).
    groups : List[List[str]]
        Member sentences per cluster, sorted by increasing distance to centroid.
    noise : None
        Placeholder for API parity with DBSCAN (which can return noise).
    meta : Dict
        Basic diagnostics: total item count, noise count (0), and cluster_num.
    """
    n_clusters = min(n_clusters, len(sentences))
    km = KMeans(n_clusters=n_clusters, n_init=500, random_state=seed)
    labels = km.fit_predict(embs)
    centers = km.cluster_centers_

    clusters = []
    for cluster_id in range(labels.max() + 1):
        idxs = np.where(labels == cluster_id)[0]
        core_idx, dists = find_core_sample(embs[idxs], core_emb=centers[cluster_id])
        group = [sentences[i] for i in idxs]
        sorted_group = [s for s, _ in sorted(zip(group, dists), key=lambda x: x[1])]
        clusters.append({"core": sentences[idxs[core_idx]], "cluster": sorted_group})

    meta = {"total": len(sentences), "noise": 0, "cluster_num": labels.max() + 1}
    cores = [c["core"] for c in clusters]
    groups = [c["cluster"] for c in clusters]
    return cores, groups, None, meta


def clustering_impl_dbscan(
    embs: np.ndarray, sentences: list[str], eps: float, min_samples: int
) -> tuple[list[str], list[list[str]], list[str], dict]:
    """Cluster sentence embeddings with DBSCAN and pick a core item per cluster.

    Returns (cores, groups, noise, meta) analogous to the K-Means variant.
    The core sample per cluster is the item closest to the cluster mean.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(embs)

    clusters = []
    for cluster_id in range(labels.max() + 1):
        idxs = np.where(labels == cluster_id)[0]
        core_idx, _ = find_core_sample(embs[idxs])
        group = [sentences[i] for i in idxs]
        clusters.append({"core": sentences[idxs[core_idx]], "cluster": group})

    noise = [sentences[i] for i, lab in enumerate(labels) if lab == -1]
    meta = {
        "total": len(sentences),
        "noise": len(noise),
        "cluster_num": labels.max() + 1,
    }
    cores = [c["core"] for c in clusters]
    groups = [c["cluster"] for c in clusters]
    return cores, groups, noise, meta


def embs_clustering(
    model: SentenceTransformer,
    sentences: list[str],
    algorithm: str,
    seed: int,
    kmeans_n_clusters: Optional[int] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None,
):
    """Encode sentences and run the requested clustering algorithm.

    Parameters allow switching between 'kmeans' and 'dbscan'. The function returns
    cluster representatives and memberships along with simple metadata used by the
    caller for graph splitting.
    """
    embs = model.encode(sentences, batch_size=32, show_progress_bar=False)
    if algorithm == "kmeans":
        return clustering_impl_kmeans(embs, sentences, seed, kmeans_n_clusters or 10)
    elif algorithm == "dbscan":
        return clustering_impl_dbscan(
            embs, sentences, dbscan_eps or 0.5, dbscan_min_samples or 5
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def graph_split_by_embs_clustering(
    graph: nx.Graph,
    model: SentenceTransformer,
    init_subgraph_count: int,
    seed: int = 42,
) -> list[nx.Graph]:
    """Split a topic graph into subgraphs using K-Means over argument embeddings.

    Nodes whose argument texts belong to the same cluster end up in the same
    subgraph. Edges are preserved within each induced subgraph.
    """
    arguments = [data["argument"] for _, data in graph.nodes(data=True)]
    _, arg_clusters, _, _ = embs_clustering(
        model=model,
        sentences=arguments,
        algorithm="kmeans",
        seed=seed,
        kmeans_n_clusters=init_subgraph_count,
    )

    subgraphs = []
    for cl in arg_clusters:
        nodes = [n for n, d in graph.nodes(data=True) if d["argument"] in cl]
        subgraphs.append(graph.subgraph(nodes).copy())
    return subgraphs


def clustering_soft(
    graph: nx.Graph,
    model: SentenceTransformer,
    init_subgraph_count: int = 10,
    max_loops: int = 100,
    seed: int = 42,
    threshold: float = 0.008,
) -> list[nx.Graph]:
    """Stochastic soft refinement of initial subgraphs (hill-climbing style).

    At each step, a random node is considered for moving into a different subgraph
    if doing so increases the average edge weight of that target subgraph. If
    removing the node from its original subgraph hurts less than `threshold`, we
    remove it; otherwise we keep it in both (soft partition). The loop is bounded by
    `max_loops` to ensure predictable runtime.
    """
    random.seed(seed)
    subgraphs = [
        g
        for g in graph_split_by_embs_clustering(graph, model, init_subgraph_count, seed)
        if g.number_of_nodes() > 1
    ]

    # Helper: compute the average weight over all non-empty KP edges in a graph.
    def avg_edge_weight(g: nx.Graph) -> float:
        weights = [
            d["weight"]
            for _, _, d in g.edges(data=True)
            if not d["kp"].startswith("No")
        ]
        return (sum(weights) / len(weights)) if weights else 0.0

    moves = 0
    for _ in tqdm(range(max_loops), desc="Hill-climbing"):
        if not subgraphs or max(len(g) for g in subgraphs) == 1:
            break

        i_out = random.choice([i for i, g in enumerate(subgraphs) if len(g) > 1])
        out_g = subgraphs[i_out]
        node = random.choice(list(out_g.nodes))
        rest_nodes = [n for n in out_g.nodes if n != node]

        base_w = avg_edge_weight(out_g)
        new_w_out = avg_edge_weight(out_g.subgraph(rest_nodes).copy())

        deltas = []
        for j, g in enumerate(subgraphs):
            if j == i_out or g.has_node(node):
                deltas.append(0)
            else:
                g_in = graph.subgraph(list(g.nodes) + [node]).copy()
                if g_in.number_of_edges() == g.number_of_edges():
                    deltas.append(0)
                else:
                    deltas.append(avg_edge_weight(g_in) - avg_edge_weight(g))

        if max(deltas) <= 0:
            continue

        target = deltas.index(max(deltas))
        subgraphs[target] = graph.subgraph(
            list(subgraphs[target].nodes) + [node]
        ).copy()
        if new_w_out - base_w > -threshold:
            # Removing the node doesn't hurt too much → move it
            subgraphs[i_out] = graph.subgraph(rest_nodes).copy()
        else:
            # Soft partition: keep the node also in its original subgraph (no deletion)
            pass

        moves += 1

    print(f"Moves executed: {moves}")
    return [g for g in subgraphs if g.number_of_nodes() > 1]


def graph_clustering(
    outputs: list[dict],
    embs_model: SentenceTransformer,
    init_subgraph_count: int = 10,
    max_loop_count: int = 200,
    seed: int = 42,
    temperature: float = 1.0,
    threshold: float = 0.008,
    target_kp_count: Optional[int] = None,
    dynamic_kp_count: bool = False,
    min_yes_prob: float = 0.0,
) -> dict[str, list[str]]:
    """Build per-topic argument graphs and extract a fixed number of keypoints.

    Steps:
    1) Construct an undirected graph per topic where nodes are unique arguments.
       Add an edge only if the generation result is affirmative (starts with "Yes").
       Use the first-token "Yes" probability as the edge weight; keep the remainder
       of the sequence as the candidate keypoint text.
    2) Split each topic graph into subgraphs via embedding-based clustering.
    3) From each subgraph, take the highest-weight edge as a keypoint candidate and
       de-duplicate similar texts; if we have fewer than `target_kp_count`, fill
       from the global top edges of the topic graph.

    We intentionally print only **aggregate** diagnostics outside of tight loops.
    """
    topic_graph: dict[str, nx.Graph] = {}
    topic_args: dict[str, list[str]] = {}
    # Aggregation containers for end-of-function summary (no per-item logging)
    sub_counts: list[int] = []
    final_kp_counts: list[int] = []

    for d in outputs:
        topic, a1, a2 = (s.strip() for s in d["prompt"].split("|"))
        topic = normalize_topic_graph(topic)
        arg_a = a1.split(".", 1)[1].strip()
        arg_b = a2.split(".", 1)[1].strip()

        G = topic_graph.setdefault(topic, nx.Graph())
        args = topic_args.setdefault(topic, [])
        for arg in (arg_a, arg_b):
            if arg not in args:
                args.append(arg)
                G.add_node(args.index(arg), argument=arg)

        ia, ib = args.index(arg_a), args.index(arg_b)

        # --- Use only affirmative ("Yes") pairs to create edges ---
        pred_text = str(d.get("pred", "")).strip()
        max_tok = str(d.get("max_token", ""))
        yes_prob = float(d.get("confidence_score", 0.0))

        # Robustly detect "Yes" at the beginning (e.g., "Yes.", "Yes:", "Yes -")
        is_affirmative = bool(re.match(r"^\s*Yes\b", pred_text)) or max_tok in {
            "▁Yes",
            "Yes",
        }
        # Paper keeps edges whenever the model outputs "Yes"; allow optional threshold via min_yes_prob
        if not is_affirmative or yes_prob < min_yes_prob:
            continue

        # Edge weight from P(Yes) (Eq. 1), optionally smoothed with raw logits
        weight = yes_prob
        raw = d.get("raw_scores", None)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            p_yes, p_no = float(raw[0]), float(raw[1])
            w_soft = math.exp(p_yes / temperature) / (
                math.exp(p_yes / temperature) + math.exp(p_no / temperature)
            )
            weight = (weight + float(w_soft)) / 2.0

        # Remove the leading label token ("Yes" + punctuation) to keep only the KP text.
        # Extract key point text after "Yes" + punctuation and normalize
        kp_text = re.sub(r"^\s*Yes\s*[\.:,\-\u2013\u2014]*\s*", "", pred_text).strip()
        # Guard: ignore degenerate or empty kp
        if not kp_text:
            continue

        if not G.has_edge(ia, ib):
            G.add_edge(ia, ib, weight=weight, kp=kp_text)

    topic_to_keypoints: dict[str, list[str]] = {}
    for topic, G in topic_graph.items():
        subs = clustering_soft(
            G, embs_model, init_subgraph_count, max_loop_count, seed, threshold
        )
        # Track number of subgraphs produced for this topic (for summary only).
        sub_counts.append(len(subs))
        # 1) One KP per subgraph (best edge)
        kp_items = []  # list of tuples (kp_text, weight)
        for sg in subs:
            if sg.number_of_edges() > 0:
                best_edge = max(sg.edges(data=True), key=lambda e: e[2]["weight"])
                kp_items.append(
                    (str(best_edge[2]["kp"]).strip(), float(best_edge[2]["weight"]))
                )

        # --- De-duplicate early (paper outputs are a concise set) ---
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().rstrip(".").lower())

        deduped = []
        seen = set()
        for text, w in sorted(kp_items, key=lambda x: x[1], reverse=True):
            key = _norm(text)
            if key and key not in seen:
                deduped.append((text, w))
                seen.add(key)
        kp_items = deduped

        # Filling and padding logic
        if dynamic_kp_count:
            # Paper mode: one KP per subgraph, no fill-up or padding
            cand_kps = [kp for kp, _ in kp_items]
        else:
            # 2) If fewer than target_kp_count, fill from global top edges of the topic graph
            if target_kp_count is None:
                tkc = 10
            else:
                tkc = target_kp_count
            if len(kp_items) < tkc:
                all_edges_sorted = sorted(
                    G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True
                )
                for u, v, data in all_edges_sorted:
                    kp_txt = str(data.get("kp", "")).strip()
                    key = _norm(kp_txt)
                    if not key or key in seen:
                        continue
                    kp_items.append((kp_txt, float(data.get("weight", 0.0))))
                    seen.add(key)
                    if len(kp_items) >= tkc:
                        break
            # 3) Finalize exactly target_kp_count outputs (pad with empty strings)
            kp_items = sorted(kp_items, key=lambda x: x[1], reverse=True)
            cand_kps = [k for k, _ in kp_items[:tkc]]
            if len(cand_kps) < tkc:
                cand_kps.extend([""] * (tkc - len(cand_kps)))

        # Count non-empty keypoints for this topic (for summary only).
        final_kp_counts.append(sum(1 for k in cand_kps if k))

        topic_to_keypoints[topic] = cand_kps

    # ---- Graph/KP summary (concise; aggregates only) ----
    def _min_avg_max(seq):
        return (
            (min(seq) if seq else 0),
            (sum(seq) / len(seq) if seq else 0.0),
            (max(seq) if seq else 0),
        )

    num_topics = len(topic_graph)
    nodes_per_topic = [g.number_of_nodes() for g in topic_graph.values()]
    edges_per_topic = [g.number_of_edges() for g in topic_graph.values()]
    all_weights = [
        float(d.get("weight", 0.0))
        for G in topic_graph.values()
        for _, _, d in G.edges(data=True)
    ]

    n_min, n_avg, n_max = _min_avg_max(nodes_per_topic)
    e_min, e_avg, e_max = _min_avg_max(edges_per_topic)
    if all_weights:
        w_min, w_avg, w_max = (
            min(all_weights),
            sum(all_weights) / len(all_weights),
            max(all_weights),
        )
    else:
        w_min = w_avg = w_max = 0.0
    s_min, s_avg, s_max = _min_avg_max(sub_counts)
    k_min, k_avg, k_max = _min_avg_max(final_kp_counts)

    print(
        f"[SUMMARY][Graph] topics={num_topics}, nodes/topic: min={n_min}, avg={n_avg:.2f}, max={n_max}; "
        f"edges/topic: min={e_min}, avg={e_avg:.2f}, max={e_max}; edge weights: min={w_min:.4f}, avg={w_avg:.4f}, max={w_max:.4f}"
    )
    print(
        f"[SUMMARY][Graph] subgraphs/topic (post-refinement): min={s_min}, avg={s_avg:.2f}, max={s_max}; "
        f"final keypoints/topic (non-empty): min={k_min}, avg={k_avg:.2f}, max={k_max}"
    )

    return topic_to_keypoints


def run_graph_phase(args, generation_results: list[dict]) -> dict[str, list[str]]:
    print("Loading embedding model …")
    embs_model = SentenceTransformer(args.embs_model_path)

    print("Running graph clustering …")
    # Note: detailed per-topic logs are avoided; a concise aggregate summary is printed inside graph_clustering().
    keypoints = graph_clustering(
        outputs=generation_results,
        embs_model=embs_model,
        init_subgraph_count=args.init_subgraph_count,
        max_loop_count=args.max_loop_count,
        seed=args.seed,
        temperature=args.temperature,
        threshold=args.threshold,
        target_kp_count=args.target_kp_count,
        dynamic_kp_count=args.dynamic_kp_count,
        min_yes_prob=args.min_yes_prob,
    )

    # Write keypoints to CSV
    rows = []
    id_counter = 1
    for topic, kps in keypoints.items():
        for kp in kps:
            kp_clean = (kp or "").strip()
            if not kp_clean:
                continue
            rows.append(
                {"key_point_id": id_counter, "topic": topic, "key_point": kp_clean}
            )
            id_counter += 1
    df_kp = pd.DataFrame(rows)
    df_kp.to_csv(args.kp_out_file, index=False, encoding="utf8")

    print(f"Finished! Keypoints CSV saved to: {args.kp_out_file}")
    return keypoints


# -----------------------------
# ------------- CLI -----------
# -----------------------------


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline: Argument-Paar-Generierung (Skript 1) -> Graph-Keypoint-Clustering (Skript 2)"
    )

    # Phase 1
    parser.add_argument(
        "--seq2seq_model_path",
        default="./models/roberta_large",
        help="Path to a (fine-tuned) Flan-T5 model as in the paper.",
    )
    parser.add_argument("--input_csv", default="../data/ines/all_arguments.csv")
    parser.add_argument("--input_json", default="./output/reformatted_input.json")
    parser.add_argument("--output_json", default="./output/reformatted_output.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--cache_path",
        type=str,
        default="./output/generate_cache.jsonl",
        help="(Deprecated) Explicit cache file path. If provided, overrides the per-dataset cache resolution.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./output/cache",
        help="Directory root for caching. A subfolder will be created per dataset+model+gen-params.",
    )
    parser.add_argument(
        "--cache_namespace",
        type=str,
        default="",
        help="Optional extra directory level to separate experiments, e.g. 'ablation1'.",
    )
    parser.add_argument(
        "--min_yes_prob",
        type=float,
        default=0.0,
        help='Keep an edge only if P(Yes) >= this value. Use 0.0 to match the paper (keep all "Yes").',
    )

    # Phase 2
    parser.add_argument("--embs_model_path", default="./models/bge-large-en-v1.5")
    parser.add_argument("--init_subgraph_count", type=int, default=10)
    parser.add_argument("--max_loop_count", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.008)
    parser.add_argument("--kp_out_file", default="./output/li_keypoints_Ines.csv")
    parser.add_argument(
        "--target_kp_count",
        type=int,
        default=10,
        help="Exact number of keypoints to output per topic (paper-style s).",
    )
    parser.add_argument(
        "--dynamic_kp_count",
        action="store_true",
        help="Output one keypoint per subgraph without fill-up/padding (as in the paper).",
    )

    # Shared
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topicwise",
        action="store_true",
        help="Process each topic separately; creates intermediate files per topic to lower memory usage.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Configuration summary ----
    print(
        "[CONFIG] device={d}, seq2seq_model='{m1}', embs_model='{m2}', "
        "batch_size={bs}, max_input_length={mil}, max_new_tokens={mnt}, "
        "target_kp_count={tkp}, init_subgraph_count={isc}, max_loop_count={mlc}, "
        "temperature={temp}, threshold={thr}, min_yes_prob={myp}, dynamic_count={dct}, topicwise={tw}".format(
            d=device.type,
            m1=args.seq2seq_model_path,
            m2=args.embs_model_path,
            bs=args.batch_size,
            mil=args.max_input_length,
            mnt=args.max_new_tokens,
            tkp=args.target_kp_count,
            isc=args.init_subgraph_count,
            mlc=args.max_loop_count,
            temp=args.temperature,
            thr=args.threshold,
            myp=args.min_yes_prob,
            dct=args.dynamic_kp_count,
            tw=args.topicwise,
        )
    )

    # ------------------------------------------------------------------
    # TOPIC‑WISE OR GLOBAL EXECUTION
    # ------------------------------------------------------------------
    if args.topicwise:
        df_all = pd.read_csv(args.input_csv)
        all_rows = []

        for topic, df_topic in df_all.groupby("topic", sort=False):
            topic_slug = _slugify(topic)
            tmp_csv = Path("./tmp") / f"{topic_slug}_arguments.csv"
            tmp_csv.parent.mkdir(parents=True, exist_ok=True)
            df_topic.to_csv(tmp_csv, index=False, encoding="utf8")

            # Clone the parsed arguments and adapt file paths per topic
            args_t = copy.deepcopy(args)
            args_t.input_csv = str(tmp_csv)
            args_t.input_json = f"./output/{topic_slug}_input.json"
            args_t.output_json = f"./output/{topic_slug}_output.json"
            args_t.kp_out_file = f"./output/keypoints_{topic_slug}.csv"

            print(f"[TOPICWISE] Processing '{topic}' with {len(df_topic)} arguments")
            gen_results = run_generation_phase(args_t, device)
            topic_kps = run_graph_phase(args_t, gen_results)

            for tp, kps in topic_kps.items():
                for kp in kps:
                    if kp.strip():
                        all_rows.append({"topic": tp, "key_point": kp})

        # Aggregate all topicwise keypoints into the final CSV requested by the user
        pd.DataFrame(all_rows).to_csv(args.kp_out_file, index=False, encoding="utf8")
        print(f"[TOPICWISE] Aggregated keypoints CSV saved to: {args.kp_out_file}")
    else:
        # Original single‑pass behaviour
        generation_results = run_generation_phase(args, device)
        run_graph_phase(args, generation_results)


if __name__ == "__main__":
    main()
