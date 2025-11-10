#!/usr/bin/env python3
"""
keypoints_from_clusters.py

Reads a clusters JSONL (produced by cluster_args.py), runs a BERT pair-classifier
to compute coverage and select key points per topic.

Default method = v11 (spaCy sentence splitting + (coverage^5)/len scoring)

Example:
    python keypoints_from_clusters.py \
        --clusters clusters.jsonl \
        --bert-model ./models/2 \
        --method v11 \
        --out keypoints.jsonl

Optional:
    --out-csv keypoints.csv
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import softmax

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

from transformers import BertForSequenceClassification, BertTokenizer

# --------- BERT helper stuff ----------

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    return AdamW(optimizer_grouped_parameters, lr=2e-5)

def ds(pairs):
    tokenizer = get_tokenizer()
    token_ids, mask_ids, seg_ids = [], [], []
    for prem, hyp in pairs:
        prem_ids = tokenizer.encode(prem, add_special_tokens=False)
        hyp_ids = tokenizer.encode(hyp, add_special_tokens=False)
        pair_ids = [tokenizer.cls_token_id] + prem_ids + [tokenizer.sep_token_id] + hyp_ids + [tokenizer.sep_token_id]
        seg = torch.tensor([0]*(len(prem_ids)+2) + [1]*(len(hyp_ids)+1))
        mask = torch.tensor([1]*len(pair_ids))
        token_ids.append(torch.tensor(pair_ids))
        seg_ids.append(seg)
        mask_ids.append(mask)

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids   = pad_sequence(seg_ids, batch_first=True)
    return TensorDataset(token_ids, mask_ids, seg_ids)

def get_preds(loader, model, device):
    optimizer = get_optimizer(model)
    preds = []
    for token_ids, mask_ids, seg_ids in tqdm(loader, desc="Batches", leave=False):
        with torch.no_grad():
            optimizer.zero_grad()
            token_ids = token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids  = seg_ids.to(device)
            out = model(token_ids, mask_ids, seg_ids)
            preds.extend(out[0].tolist())
    return preds

def arguments_coverage_all_candidates(arguments_list, model, device):
    pairs = [[a, b] for a in arguments_list for b in arguments_list]
    dataset = ds(pairs)
    loader = DataLoader(dataset, shuffle=False, batch_size=128)
    raw_preds = get_preds(loader, model, device)
    labeled = [1 if p[1] > p[0] else 0 for p in raw_preds]

    coverage = {}
    n = len(arguments_list)
    for i, arg in enumerate(arguments_list):
        start, end = i*n, (i+1)*n
        coverage[arg] = sum(labeled[start:end])
    return coverage

# --------- selection utilities ----------

def sort_sums(cluster_summary_by_topic):
    """Return only the ordered texts per topic by decreasing coverage."""
    sorted_summaries = []
    for topic in cluster_summary_by_topic:
        pairs = list(cluster_summary_by_topic[topic])
        pairs.sort(reverse=True, key=lambda x: x[1])
        sorted_summaries.append([text for text,_ in pairs])
    return sorted_summaries

def top_arg_shortest(arg_cov):
    max_cov = max(arg_cov.values()) if arg_cov else 0
    winners = [a for a, cov in arg_cov.items() if cov == max_cov]
    if not winners:
        return ""
    return sorted(winners, key=len)[0]

def top_arg_scoring_v1(arg_cov):
    best_score = -1
    best_args = []
    for arg, cov in arg_cov.items():
        if not arg.strip():
            continue
        score = (cov ** 5) / max(1, len(arg.split()))
        if score > best_score:
            best_score = score
            best_args = [arg]
        elif score == best_score:
            best_args.append(arg)
    if not best_args:
        return ""
    return sorted(best_args, key=len)[0]

def method_v6(cluster_by_topic, model, device):
    """Per cluster: split by '.', choose shortest among top coverage."""
    summary_by_topic = {}
    for topic, clusters in cluster_by_topic.items():
        out = []
        for _, args in clusters.items():
            sentences = [sent for a in args for sent in a.split('.')]
            sentences = [s.strip() for s in sentences if s.strip()]
            cov = arguments_coverage_all_candidates(sentences, model, device)
            best = top_arg_shortest(cov)
            out.append([best, len(sentences)])
        summary_by_topic[topic] = out
    return sort_sums(summary_by_topic)

def method_v11(cluster_by_topic, model, device):
    """Per cluster: split with spaCy, choose best scoring (coverage^5 / len)."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy is required for method_v11") from e

    summary_by_topic = {}
    for topic, clusters in cluster_by_topic.items():
        out = []
        for _, args in clusters.items():
            sents = []
            for a in args:
                sents.extend(str(s) for s in nlp(a).sents)
            sents = [s.strip() for s in sents if s.strip()]
            cov = arguments_coverage_all_candidates(sents, model, device)
            best = top_arg_scoring_v1(cov)
            out.append([best, len(sents)])
        summary_by_topic[topic] = out
    return sort_sums(summary_by_topic)

# ------------- main -------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", required=True, help="JSONL produced by cluster_args.py")
    parser.add_argument("--bert-model", required=True, help="Path/name for fine-tuned BertForSequenceClassification")
    parser.add_argument("--method", choices=["v6", "v11"], default="v11")
    parser.add_argument("--limit", type=int, default=None,
                        help="Return at most N keypoints per topic (after sorting)")
    parser.add_argument("--out", required=True, help="Output JSONL with keypoints per topic")
    parser.add_argument("--out-csv", default=None, help="Optional: also write CSV with columns=topics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)
    model.eval()

    # Load clusters
    cluster_by_topic = {}
    with open(args.clusters, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            topic = rec["topic"]
            # keys back to int
            clusters = {int(k): v for k, v in rec["clusters"].items()}
            cluster_by_topic[topic] = clusters

    # Run summarization
    start = time.time()
    if args.method == "v6":
        summaries = method_v6(cluster_by_topic, model, device)
    else:
        summaries = method_v11(cluster_by_topic, model, device)
    print(f"Summaries computed in {time.time()-start:.1f}s")

    topics = list(cluster_by_topic.keys())

    # Apply optional limit
    if args.limit is not None:
        summaries = [kp_list[:args.limit] for kp_list in summaries]

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for topic, kp_list in zip(topics, summaries):
            f.write(json.dumps({"topic": topic, "keypoints": kp_list}, ensure_ascii=False) + "\n")

    # Optional CSV (columns = topics)
    if args.out_csv:
        df = pd.DataFrame({t: summaries[i] for i, t in enumerate(topics)})
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote CSV to {args.out_csv}")

    print(f"Wrote JSONL to {args.out}")


if __name__ == "__main__":
    main()