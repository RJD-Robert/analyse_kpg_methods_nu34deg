# Standard Library
import copy
from collections import defaultdict
import time

# Third-Party Libraries
import numpy as np
import pandas as pd
from scipy.special import softmax
import spacy
from tqdm.auto import tqdm

# PyTorch
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# Transformers & Sentence Transformers
from transformers import BertForSequenceClassification, BertTokenizer
from sentence_transformers import SentenceTransformer

# Sklearn clustering
from sklearn.cluster import AgglomerativeClustering

# Ensure spaCy model is available
import spacy
spacy.cli.download("en_core_web_sm")


# -------------------
# Global configuration
# -------------------
arguments_file = "../data/shared_task_data/arguments_combined.csv"
embedder = SentenceTransformer('./models/V1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = BertForSequenceClassification.from_pretrained("./models/2", num_labels=2)
model.to(device)

NUM_KEYPOINTS = 10  # maximum clusters per topic


# --------------------------------
# Helper functions for tokenization,
# optimization, and prediction
# --------------------------------

def get_tokenizer():
    """Load the BERT tokenizer (uncased)."""
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_optimizer(model):
    """Set up AdamW optimizer with weight-decay settings."""
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0
        }
    ]
    return AdamW(optimizer_grouped_parameters, lr=2e-5)

def get_preds(test_loader, model):
    """Run the model on all batches and collect raw logits."""
    optimizer = get_optimizer(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for token_ids, mask_ids, seg_ids in tqdm(test_loader, desc="Batches", leave=False):
        with torch.no_grad():
            optimizer.zero_grad()
            token_ids = token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            out = model(token_ids, mask_ids, seg_ids)
            preds.extend(out[0].tolist())
    return preds

def ds(premise_hyp_pairs):
    """Convert list of (premise, hypothesis) into a TensorDataset."""
    tokenizer = get_tokenizer()
    token_ids, mask_ids, seg_ids = [], [], []
    for prem, hyp in premise_hyp_pairs:
        # encode without special tokens, then add [CLS], [SEP], [SEP]
        prem_ids = tokenizer.encode(prem, add_special_tokens=False)
        hyp_ids = tokenizer.encode(hyp, add_special_tokens=False)
        pair_ids = [tokenizer.cls_token_id] + prem_ids + [tokenizer.sep_token_id] + hyp_ids + [tokenizer.sep_token_id]
        seg = torch.tensor([0]*(len(prem_ids)+2) + [1]*(len(hyp_ids)+1))
        mask = torch.tensor([1]*len(pair_ids))
        token_ids.append(torch.tensor(pair_ids))
        seg_ids.append(seg)
        mask_ids.append(mask)

    # pad sequences to same length
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids   = pad_sequence(seg_ids, batch_first=True)
    return TensorDataset(token_ids, mask_ids, seg_ids)


# ------------------------
# Summarization utilities
# ------------------------

def sort_sums(cluster_summary_by_topic):
    """Sort summaries in descending order of their coverage."""
    sorted_summaries = []
    for topic in cluster_summary_by_topic:
        pairs = list(cluster_summary_by_topic[topic])
        pairs.sort(reverse=True, key=lambda x: x[1])  # sort by coverage count
        sorted_summaries.append([text for text,_ in pairs])
    return sorted_summaries

def top_arg_shortest(topic, arg_cov):
    """Select candidate with highest coverage, then shortest length."""
    max_cov = max(arg_cov.values())
    winners = [a for a,cov in arg_cov.items() if cov == max_cov]
    return sorted(winners, key=len)[0]

def top_arg_scoring_v1(topic, arg_cov):
    """Score each argument by (coverage^5)/length, pick the best."""
    scored = {}
    best = -1
    for arg, cov in arg_cov.items():
        if not arg.strip(): 
            continue
        score = (cov**5) / len(arg.split())
        scored[arg] = score
        best = max(best, score)
    # break ties by shortest
    winners = [a for a, sc in scored.items() if sc == best]
    return sorted(winners, key=len)[0]

def arguments_coverage_all_candidates(arguments_list, model):
    """Compute how many other args each argument covers (via model predictions)."""
    pairs = [[a,b] for a in arguments_list for b in arguments_list]
    dataset = ds(pairs)
    loader = DataLoader(dataset, shuffle=False, batch_size=128)
    raw_preds = get_preds(loader, model)
    probs = softmax(raw_preds, axis=1)
    # Label a pair as covering if logit[1] > logit[0]
    labels = [1 if p[1] > p[0] else 0 for p in raw_preds]

    # Count coverage per argument
    coverage = {}
    n = len(arguments_list)
    for i, arg in enumerate(arguments_list):
        # each arg appears in n pairs
        start, end = i*n, (i+1)*n
        coverage[arg] = sum(labels[start:end])
    return coverage


# ----------------------------
# Method implementations (v6/v11)
# ----------------------------
def method_v6(cluster_by_topic, model, embedder=''):
    """Pick shortest high-coverage sentence per cluster."""
    summary_by_topic = {}
    for topic, clusters in cluster_by_topic.items():
        out = []
        for cid, args in clusters.items():
            sentences = [sent for arg in args for sent in arg.split('.')]
            cov = arguments_coverage_all_candidates(sentences, model)
            best = top_arg_shortest(topic, cov)
            out.append([best, len(sentences)])
        summary_by_topic[topic] = out
    return sort_sums(summary_by_topic)

def method_v11(cluster_by_topic, model, embedder=''):
    """Pick best-scoring sentence per cluster using spaCy sentence splits."""
    nlp = spacy.load("en_core_web_sm")
    summary_by_topic = {}
    for topic, clusters in cluster_by_topic.items():
        out = []
        for cid, args in clusters.items():
            sentences = [str(sent) for arg in args for sent in nlp(arg).sents]
            sentences = [s for s in sentences if s]
            cov = arguments_coverage_all_candidates(sentences, model)
            best = top_arg_scoring_v1(topic, cov)
            out.append([best, len(sentences)])
        summary_by_topic[topic] = out
    return sort_sums(summary_by_topic)


# -------------------------------------
# CLUSTERING: implementation and usage
# -------------------------------------
def cluster(input_args_kp_by_topic, embedder, limits=None):
    """
    Perform Agglomerative Clustering on each topic’s arguments.
    Returns:
      - cluster_by_topic: dict[topic][cluster_id] → list of argument strings
      - placeholder ''
    """
    cluster_by_topic = {}
    topics = list(input_args_kp_by_topic.keys())

    # ──────────────────────────────────────────────────
    # This loop is where the clustering happens:
    # For each topic:
    #   • embed all unique arguments with SentenceTransformer
    #   • normalize embeddings
    #   • choose # of clusters (fixed or distance-threshold)
    #   • fit AgglomerativeClustering → cluster_assignment
    #   • group sentences by cluster label
    # ──────────────────────────────────────────────────
    for topic in tqdm(topics, desc="Clustering topics"):
        # collect all candidate sentences for this topic
        arguments = list(set().union(*input_args_kp_by_topic[topic].values()))

        # embed & normalize
        corpus_embeddings = embedder.encode(arguments)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # decide n_clusters vs threshold
        if limits:
            n_clusters = limits[topics.index(topic)]
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)

        # ─── The actual clustering call ───
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        # ─────────────────────────────────

        # collect sentences per cluster
        topic_clusters = defaultdict(list)
        for sent_id, cl_id in enumerate(cluster_assignment):
            topic_clusters[cl_id].append(arguments[sent_id])
        cluster_by_topic[topic] = topic_clusters

    return cluster_by_topic, ''


def build_topic_args_dict(path_to_arguments_csv: str):
    """
    Read CSV with columns ['topic', 'argument'] → dict[topic]['ALL'] = [arg1, arg2, …]
    """
    df = pd.read_csv(path_to_arguments_csv)
    topic_args = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        topic_args[row["topic"]]["ALL"].append(row["argument"])
    return topic_args


def run_experiment_args_only(model, embedder, topic_args_dict, method):
    """
    1. Cluster the arguments   ← **Here** you call `cluster(...)`
    2. Generate summaries via chosen method (v6 or v11)
    """
    t0_total = time.time()

    print("=== Clustering arguments ===")
    # ────────────────────────────────────────────────
    # This is where you **invoke** clustering:
    clusters, _ = cluster(topic_args_dict,
                          embedder,
                          limits=[NUM_KEYPOINTS]*len(topic_args_dict))
    # ────────────────────────────────────────────────
    print(f"   ↳ Done after {time.time() - t0_total:.1f}s\n")

    print("=== Generating summaries ===")
    t0_sum = time.time()
    summaries = method(clusters, model, embedder)
    print(f"   ↳ Done after {time.time() - t0_sum:.1f}s")

    print(f"=== Total time: {(time.time() - t0_total)/60:.1f} min ===")
    return summaries


# -------------------
# Main execution block
# -------------------
print("=== Running experiment with arguments only ===")
topic_args_only = build_topic_args_dict(arguments_file)
summaries = run_experiment_args_only(model, embedder, topic_args_only, method_v11)

# Print results and save to CSV
for topic_id, topic in enumerate(topic_args_only):
    print(f"\n=== {topic} ===")
    for kp in summaries[topic_id]:
        print("-", kp)

kp_df = pd.DataFrame({t: summaries[i] for i, t in enumerate(topic_args_only.keys())})
kp_df.to_csv("khosravani_keypoints_ArgKP.csv", index=False)