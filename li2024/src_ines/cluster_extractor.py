#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import networkx as nx
import pandas as pd
import torch
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    PreTrainedTokenizer
)


# =========================
# ===== Phase 1: Gen ======
# =========================

def normalize_stance(raw: str) -> str:
    val = str(raw).strip().lower()
    if val in {"1", "+", "positive", "positiv", "pos"}:
        return "positive"
    if val in {"-1", "-", "negative", "negativ", "neg"}:
        return "negative"
    return val


def build_samples(df: pd.DataFrame) -> list[dict]:
    samples = []
    for topic, group in df.groupby("topic", sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                stance1 = normalize_stance(group.at[i, "stance"])
                stance2 = normalize_stance(group.at[j, "stance"])
                arg1 = str(group.at[i, "argument"]).strip()
                arg2 = str(group.at[j, "argument"]).strip()
                input_str = f"{topic.strip()} | {stance1}. {arg1} | {stance2}. {arg2}"
                samples.append({"id": input_str, "input": input_str})
    return samples


class DataLoaderKPA:
    def __init__(
        self,
        ids: list[str],
        queries: list[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int
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
                return_tensors="pt"
            )
            yield batch_ids, tokenized


def load_seq2seq_model(model_path: str, device: torch.device):
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
    cache_path: Path
) -> dict[str, dict]:
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
    dataloader = DataLoaderKPA(list(rem_ids), list(rem_queries), tokenizer, batch_size, max_input_length)
    model.eval()

    yes_token = tokenizer.convert_tokens_to_ids("▁Yes")
    no_token = tokenizer.convert_tokens_to_ids("▁No")

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
        decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

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
            cache_file.write(json.dumps({"id": sample_id, "data": data}, ensure_ascii=False) + "\n")

    cache_file.close()
    return outputs


def run_generation_phase(args, device: torch.device) -> list[dict]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Using device: {device}")

    df = pd.read_csv(args.input_csv)
    required = {"topic", "argument", "stance"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    samples = build_samples(df)
    Path(args.input_json).write_text(json.dumps(samples, ensure_ascii=False, indent=4))
    print(f"{len(samples)} samples written to {args.input_json}")

    model, tokenizer = load_seq2seq_model(args.seq2seq_model_path, device)
    print(f"Seq2Seq model loaded from {args.seq2seq_model_path}")

    ids = [s["id"] for s in samples]
    queries = [s["input"] for s in samples]
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        ids=ids,
        queries=queries,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        cache_path=Path(args.cache_path),
    )

    results = []
    for s in samples:
        data = outputs[s["id"]]
        results.append({
            "prompt": s["input"],
            "pred": data["output"],
            "confidence_score": data["confidence_score"],
            "max_token": data["max_token"],
            "raw_pred": data["output"],
            "raw_scores": data["raw_scores"],
        })

    Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=4))
    print(f"Results written to {args.output_json}")

    return results


# =========================
# === Phase 2: Cluster ====
# =========================

def normalize_topic_graph(text: str) -> str:
    return text.rstrip(' .') + '.'


def find_core_sample(
    embeddings: np.ndarray,
    core_emb: Optional[np.ndarray] = None
) -> tuple[int, list[float]]:
    if core_emb is None:
        core_emb = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - core_emb, axis=1)
    return int(distances.argmin()), distances.tolist()


def clustering_impl_kmeans(
    embs: np.ndarray,
    sentences: list[str],
    seed: int,
    n_clusters: int
) -> tuple[list[str], list[list[str]], None, dict]:
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
        clusters.append({
            'core': sentences[idxs[core_idx]],
            'cluster': sorted_group
        })

    meta = {
        'total': len(sentences),
        'noise': 0,
        'cluster_num': labels.max() + 1
    }
    cores = [c['core'] for c in clusters]
    groups = [c['cluster'] for c in clusters]
    return cores, groups, None, meta


def clustering_impl_dbscan(
    embs: np.ndarray,
    sentences: list[str],
    eps: float,
    min_samples: int
) -> tuple[list[str], list[list[str]], list[str], dict]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(embs)

    clusters = []
    for cluster_id in range(labels.max() + 1):
        idxs = np.where(labels == cluster_id)[0]
        core_idx, _ = find_core_sample(embs[idxs])
        group = [sentences[i] for i in idxs]
        clusters.append({
            'core': sentences[idxs[core_idx]],
            'cluster': group
        })

    noise = [sentences[i] for i, lab in enumerate(labels) if lab == -1]
    meta = {
        'total': len(sentences),
        'noise': len(noise),
        'cluster_num': labels.max() + 1
    }
    cores = [c['core'] for c in clusters]
    groups = [c['cluster'] for c in clusters]
    return cores, groups, noise, meta


def embs_clustering(
    model: SentenceTransformer,
    sentences: list[str],
    algorithm: str,
    seed: int,
    kmeans_n_clusters: Optional[int] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None
):
    embs = model.encode(sentences, batch_size=32, show_progress_bar=False)
    if algorithm == 'kmeans':
        return clustering_impl_kmeans(embs, sentences, seed, kmeans_n_clusters or 10)
    elif algorithm == 'dbscan':
        return clustering_impl_dbscan(embs, sentences, dbscan_eps or 0.5, dbscan_min_samples or 5)
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')


def graph_split_by_embs_clustering(
    graph: nx.Graph,
    model: SentenceTransformer,
    init_subgraph_count: int,
    seed: int = 42
) -> list[nx.Graph]:
    arguments = [data['argument'] for _, data in graph.nodes(data=True)]
    _, arg_clusters, _, _ = embs_clustering(
        model=model,
        sentences=arguments,
        algorithm='kmeans',
        seed=seed,
        kmeans_n_clusters=init_subgraph_count
    )

    subgraphs = []
    for cl in arg_clusters:
        nodes = [n for n, d in graph.nodes(data=True) if d['argument'] in cl]
        subgraphs.append(graph.subgraph(nodes).copy())
    return subgraphs


def clustering_soft(
    graph: nx.Graph,
    model: SentenceTransformer,
    init_subgraph_count: int = 10,
    max_loops: int = 100,
    seed: int = 42,
    threshold: float = 0.008
) -> list[nx.Graph]:
    random.seed(seed)
    subgraphs = [
        g for g in graph_split_by_embs_clustering(
            graph, model, init_subgraph_count, seed
        ) if g.number_of_nodes() > 1
    ]

    def avg_edge_weight(g: nx.Graph) -> float:
        weights = [d['weight'] for _, _, d in g.edges(data=True) if not d['kp'].startswith('No')]
        return (sum(weights) / len(g.edges)) if g.edges else 0.0

    moves = 0
    for _ in tqdm(range(max_loops), desc='Hill-climbing'):
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
        subgraphs[target] = graph.subgraph(list(subgraphs[target].nodes) + [node]).copy()
        if new_w_out - base_w > -threshold:
            subgraphs[i_out] = graph.subgraph(rest_nodes).copy()
        else:
            subgraphs[i_out] = None

        subgraphs = [g for g in subgraphs if g is not None]
        moves += 1

    print(f'Moves executed: {moves}')
    return [g for g in subgraphs if g.number_of_nodes() > 1]


def build_topic_graphs(outputs: list[dict], temperature: float = 1.0) -> dict[str, nx.Graph]:
    """Build one graph per topic from generation outputs."""
    topic_graph: dict[str, nx.Graph] = {}
    topic_args: dict[str, list[str]] = {}

    for d in outputs:
        topic, a1, a2 = (s.strip() for s in d['prompt'].split('|'))
        topic = normalize_topic_graph(topic)
        arg_a = a1.split('.', 1)[1].strip()
        arg_b = a2.split('.', 1)[1].strip()

        G = topic_graph.setdefault(topic, nx.Graph())
        args = topic_args.setdefault(topic, [])
        for arg in (arg_a, arg_b):
            if arg not in args:
                args.append(arg)
                G.add_node(args.index(arg), argument=arg)

        ia, ib = args.index(arg_a), args.index(arg_b)
        if not G.has_edge(ia, ib):
            p_yes, p_no = d['raw_scores']
            w = math.exp(p_yes / temperature) / (math.exp(p_yes / temperature) + math.exp(p_no / temperature))
            G.add_edge(
                ia, ib,
                weight=w,
                kp=d['pred'].removeprefix('Yes.').strip(),
                answer=d.get('answer', '').removeprefix('Yes.').strip()
            )

    return topic_graph


def write_clusters_jsonl(
    topic_graphs: dict[str, nx.Graph],
    embs_model_path: str,
    clusters_out: Path,
    init_subgraph_count: int,
    max_loop_count: int,
    seed: int,
    threshold: float
):
    embs_model = SentenceTransformer(embs_model_path)
    with clusters_out.open("w", encoding="utf-8") as fout:
        for topic, G in topic_graphs.items():
            subgraphs = clustering_soft(
                G, embs_model,
                init_subgraph_count=init_subgraph_count,
                max_loops=max_loop_count,
                seed=seed,
                threshold=threshold
            )

            for cid, sg in enumerate(subgraphs):
                args = [data['argument'] for _, data in sg.nodes(data=True)]
                edges = []
                weights = []
                for u, v, d in sg.edges(data=True):
                    a = sg.nodes[u]['argument']
                    b = sg.nodes[v]['argument']
                    w = float(d['weight'])
                    edges.append({
                        "a": a,
                        "b": b,
                        "weight": w,
                        "kp": d.get('kp', ''),
                        "answer": d.get('answer', '')
                    })
                    weights.append(w)

                meta = {
                    "avg_weight": float(sum(weights) / len(weights)) if weights else 0.0,
                    "num_edges": len(edges)
                }

                fout.write(json.dumps({
                    "topic": topic,
                    "cluster_id": cid,
                    "arguments": args,
                    "edges": edges,
                    "meta": meta
                }, ensure_ascii=False) + "\n")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="(Part 1) Generate + Graph build + Cluster. Writes clusters to JSONL."
    )
    # Phase 1
    p.add_argument('--seq2seq_model_path', default="../models/roberta_large")
    p.add_argument('--input_csv', default='../../data/ines_data/20_arguments_for_every_topic.csv')
    p.add_argument('--input_json', default='./output/reformatted_input.json')
    p.add_argument('--output_json', default='./output/reformatted_output.json')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_input_length', type=int, default=512)
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--cache_path', type=str, default='./output/generate_cache.jsonl')

    # Phase 2 (clustering only, NO keypoint extraction here)
    p.add_argument('--embs_model_path', default='../models/bge-large-en-v1.5')
    p.add_argument('--init_subgraph_count', type=int, default=10)
    p.add_argument('--max_loop_count', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--threshold', type=float, default=0.008)

    # IO
    p.add_argument('--clusters_out', type=str, default='./output/clusters.jsonl')

    # Shared
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1: model inference
    generation_results = run_generation_phase(args, device)

    # Build per-topic graphs
    topic_graphs = build_topic_graphs(generation_results, temperature=args.temperature)

    # Cluster and dump clusters as JSONL
    write_clusters_jsonl(
        topic_graphs=topic_graphs,
        embs_model_path=args.embs_model_path,
        clusters_out=Path(args.clusters_out),
        init_subgraph_count=args.init_subgraph_count,
        max_loop_count=args.max_loop_count,
        seed=args.seed,
        threshold=args.threshold
    )

    print(f"Clusters written to {args.clusters_out}")


if __name__ == "__main__":
    main()