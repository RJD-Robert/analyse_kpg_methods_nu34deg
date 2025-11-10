# ==========================================
# KPG Pipeline – Added verbose logging (prints)
# ------------------------------------------
# The following print statements and comments were inserted to make the
# execution flow and intermediate results transparent for first-time readers.
# No functional logic was changed; only explanatory comments and console output
# were added. All comments are in English as requested.
# ==========================================

# Standard Library
import copy
from collections import defaultdict
import time

# Additional imports for caching/resume
import os
import pickle
import hashlib
import re

# Third Party Libraries
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

# Sklearn
from sklearn.cluster import AgglomerativeClustering

import spacy
import argparse

# ---- CLI ----


def _parse_cli():
    p = argparse.ArgumentParser(description="KPG Pipeline (CLI configurable)")
    p.add_argument(
        "--arguments-file",
        "-a",
        default="../modified_KPG/output/argmatch_processed_3x300.csv",
        help="Path to the CSV containing arguments (with columns: topic, argument)",
    )
    p.add_argument(
        "--embedder",
        default="./models/V1",
        help="Path/name of sentence-transformer model for embeddings",
    )
    p.add_argument(
        "--classifier",
        default="./models/2",
        help="Path/name of fine-tuned BERT classifier directory",
    )
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=10,
        help="Number of keypoints (clusters) per topic",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed id")
    p.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for pairwise scoring"
    )
    p.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model to use for sentence splitting",
    )
    p.add_argument(
        "--cache-dir",
        default="./output/cache_kpg",
        help="Directory for pipeline caches",
    )
    p.add_argument("--no-cache", action="store_true", help="Disable cache reads/writes")
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cache exists",
    )
    p.add_argument(
        "--log-mode",
        choices=["summary", "detailed", "none"],
        default="summary",
        help="Logging verbosity",
    )
    return p.parse_args()


ARGS = _parse_cli()

print("#" * 35)
print("USING CUDA" if torch.cuda.is_available() else "USING CPU")
print("#" * 35)


spacy.cli.download(ARGS.spacy_model)
print(f"[BOOT] Ensured spaCy model '{ARGS.spacy_model}' is available.")

arguments_file = ARGS.arguments_file
print(f"[CONFIG] Path to arguments CSV: {arguments_file}")

# ---- Logging mode & global metrics (summary-focused) ----
# We avoid per-element prints; instead, we accumulate metrics here and
# print a compact summary at the very end of the pipeline.
LOG_MODE = ARGS.log_mode  # options: "summary", "detailed", "none"
METRICS = {
    "seq_lengths": [],  # token sequence lengths across ALL ds() calls
    "pos_probs": [],  # positive-class probabilities across ALL coverage computations
    "labeled_pos": 0,  # total number of positive pair labels
    "labeled_neg": 0,  # total number of negative pair labels
    "coverage_values": [],  # per-argument coverage values across ALL topics
    "cluster_sizes": [],  # sizes of all formed clusters across topics
    "pred_vectors": 0,  # total number of prediction vectors produced
    "total_arguments": 0,  # total unique arguments considered across topics
}

print(f"[LOAD] Loading sentence embedder from '{ARGS.embedder}' ...")
_t_embed = time.time()
embedder = SentenceTransformer(ARGS.embedder)
print(f"[LOAD] Sentence embedder ready in {time.time() - _t_embed:.2f}s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    try:
        print(
            f"[DEVICE] CUDA is available. GPU count: {torch.cuda.device_count()}, current device index: {torch.cuda.current_device()}"
        )
    except Exception as _e:
        print(f"[DEVICE] CUDA available but could not query device details: {_e}")
else:
    print("[DEVICE] CUDA is NOT available. Running on CPU.")

print(f"[LOAD] Loading classification model from '{ARGS.classifier}' ...")
_t_model = time.time()
model = BertForSequenceClassification.from_pretrained(ARGS.classifier, num_labels=2)
model.to(device)
print(f"[LOAD] Model loaded and moved to {device} in {time.time() - _t_model:.2f}s")

limits_argkp = []  # unused
seed_id = ARGS.seed
method6_sums = []
method6_coverages = []
method11_sums = []
method11_coverages = []
NUM_KEYPOINTS = ARGS.num_keypoints

print("[CONFIG] Runtime parameters:")
print(f"         seed_id={seed_id}")
print(f"         NUM_KEYPOINTS={NUM_KEYPOINTS}")
print(f"         spacy_model={ARGS.spacy_model}")
print(f"         batch_size={ARGS.batch_size}")
print(
    f"         cache_dir='{ARGS.cache_dir}', cache_enabled={not ARGS.no_cache}, force_recompute={ARGS.force_recompute}"
)

# ---- Derived/configurable globals ----
SPA_MODEL_NAME = ARGS.spacy_model
BATCH_SIZE = ARGS.batch_size


# ---- Caching / Resume setup ----
CACHE_ENABLED = not ARGS.no_cache  # turn off with --no-cache
FORCE_RECOMPUTE = ARGS.force_recompute
CACHE_DIR = ARGS.cache_dir
os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_topic_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:100]


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _hash_list(strings):
    h = hashlib.sha1()
    for s in strings:
        h.update(str(s).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _run_hash():
    base = f"args={os.path.abspath(arguments_file)}|embedder=./models/V1|clf=./models/2|k={NUM_KEYPOINTS}|seed={seed_id}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]


RUN_HASH = _run_hash()
CACHE_BASE = os.path.join(CACHE_DIR, RUN_HASH)
os.makedirs(CACHE_BASE, exist_ok=True)
print(f"[CACHE] Cache directory: {CACHE_BASE}")


# Helper to load and return a BERT tokenizer.
def get_tokenizer():
    print("[TOKENIZER] Loading BERT tokenizer 'bert-base-uncased' (lowercase=True)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    print("[TOKENIZER] Tokenizer loaded.")
    return tokenizer


# Helper to create an AdamW optimizer with weight decay for the model.
def get_optimizer(model):
    print("[OPTIM] Initializing AdamW optimizer with weight decay.")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[OPTIM] Model parameters: total={total_params:,}, trainable={trainable_params:,}"
        )
    except Exception as _e:
        print(f"[OPTIM] Could not compute parameter counts: {_e}")
    return optimizer


# Runs the model on batches of tokenized input data to get predictions.
def get_preds(test_loader, model):
    _t_pred_total = time.time()
    optimizer = get_optimizer(model)
    preds = []

    for pair_token_ids, mask_ids, seg_ids in tqdm(
        test_loader, desc="Batches", leave=False
    ):
        with torch.no_grad():
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            out = model(pair_token_ids, mask_ids, seg_ids)
            preds.extend(out[0].tolist())
    # Summary-mode: only accumulate counts; no per-batch prints
    try:
        METRICS["pred_vectors"] += len(preds)
    except Exception:
        pass
    return preds


# Converts text pairs into a TensorDataset suitable for BERT input.
def ds(data):
    _t_ds = time.time()
    lengths = []  # track pair token lengths for simple stats
    tokenizer = get_tokenizer()
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    for prem, hyp in data:
        encoding = tokenizer.encode_plus(
            prem,
            hyp,
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation="longest_first",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        pair_token_ids = encoding["input_ids"]
        attention_mask_ids = encoding["attention_mask"]
        segment_ids = encoding["token_type_ids"]
        lengths.append(len(pair_token_ids))

        token_ids.append(torch.tensor(pair_token_ids))
        mask_ids.append(torch.tensor(attention_mask_ids))
        seg_ids.append(torch.tensor(segment_ids))

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    # Summary-mode: collect all token sequence lengths across runs
    try:
        METRICS["seq_lengths"].extend(lengths)
    except Exception:
        pass
    test_ds = TensorDataset(token_ids, mask_ids, seg_ids)
    return test_ds


# Sorts summaries within each topic by their scores in descending order.
def sort_sums(cluster_summary_by_topic):
    sorted_summaries = []
    for t in cluster_summary_by_topic:
        s = list(cluster_summary_by_topic[t])
        s.sort(reverse=True, key=lambda x: x[1])
        ss = [item[0] for item in s]
        sorted_summaries.append(ss)
    return sorted_summaries


# Selects the shortest argument among those with highest coverage.
def top_arg_shortest(topic, arg_cov):
    highest = max(arg_cov.values())
    candidates = [i for i in arg_cov if arg_cov[i] == highest]
    return sorted(candidates, key=len)[0]


# Scores arguments by coverage^5 divided by length and selects the best.
def top_arg_scoring_v1(topic, arg_cov):
    scored_candids = defaultdict()
    highes_score = -1
    for arg in arg_cov:
        if not arg or len(arg.split()) == 0:
            continue
        score = (arg_cov[arg] ** 5) / len(arg.split())
        scored_candids[arg] = score
        highes_score = max(highes_score, score)
    candidates = [i for i in scored_candids if scored_candids[i] == highes_score]
    return sorted(candidates, key=len)[0]


# Computes coverage scores for all candidate arguments by pairwise model predictions.
def arguments_coverage_all_candidates(arguments_list, model):
    _t_cov = time.time()

    # --- Cache hit? ---
    cache_key = _hash_list(arguments_list) + f"_n{len(arguments_list)}"
    cov_cache_path = os.path.join(CACHE_BASE, f"cov_{cache_key}.pkl")
    if CACHE_ENABLED and not FORCE_RECOMPUTE and os.path.exists(cov_cache_path):
        try:
            arg_coverage = _load(cov_cache_path)
            return arg_coverage
        except Exception as _e:
            print(
                f"[CACHE] Failed to load coverage cache {cov_cache_path}: {_e}. Recomputing…"
            )

    # --- Compute ---
    arg_pairs = [[a, b] for idx, a in enumerate(arguments_list) for b in arguments_list]
    dsa = ds(arg_pairs)
    loader = DataLoader(dsa, shuffle=False, batch_size=BATCH_SIZE)
    preds = get_preds(loader, model)

    # Convert logits to probabilities for analysis only (no logic change to labeling below)
    pos_probs = None
    try:
        _soft = softmax(preds, axis=1)
        pos_probs = _soft[:, 1]
        try:
            # Accumulate probabilities across all coverage computations
            METRICS["pos_probs"].extend(pos_probs.tolist())
        except Exception:
            pass
    except Exception as _e:
        pass

    # We still compute softmax to stay consistent with the original pipeline
    soft_preds = softmax(preds, axis=1)
    labeled_preds = []
    for i in range(len(soft_preds)):
        if preds[i][1] > preds[i][0] and preds[i][1] > 0:
            labeled_preds.append(1)
        else:
            labeled_preds.append(0)

    try:
        ones = sum(1 for x in labeled_preds if x == 1)
        zeros = sum(1 for x in labeled_preds if x == 0)
        METRICS["labeled_pos"] += ones
        METRICS["labeled_neg"] += zeros
    except Exception:
        pass

    arg_coverage = defaultdict(dict)
    for i in range(len(arguments_list)):
        s = 0
        s1 = labeled_preds[i * len(arguments_list) : (i + 1) * len(arguments_list)]
        for ss in s1:
            s += ss
        arg_coverage[arguments_list[i]] = s

    # Accumulate per-argument coverage (quality proxy) values
    try:
        METRICS["coverage_values"].extend(list(arg_coverage.values()))
    except Exception:
        pass

    # --- Save to cache ---
    if CACHE_ENABLED:
        try:
            _dump(arg_coverage, cov_cache_path)
        except Exception as _e:
            print(f"[CACHE] Could not write coverage cache {cov_cache_path}: {_e}")

    return arg_coverage


# Summarizes clusters by selecting key arguments using method version 6.
def method_v6(cluster_by_topic, model, embedder=""):
    cluster_summary_by_topic = {}
    for topic in cluster_by_topic:
        summaries = []
        for cluster in cluster_by_topic[topic]:
            arguments = cluster_by_topic[topic][cluster]
            arguments = [
                sent for arguments in arguments for sent in arguments.split(".")
            ]
            arg_and_cov = arguments_coverage_all_candidates(arguments, model)
            sum = top_arg_shortest(topic, arg_and_cov)
            summaries.append([sum, len(arguments)])

        cluster_summary_by_topic[topic] = summaries
    sorted_summaries = sort_sums(cluster_summary_by_topic)
    return sorted_summaries


# Summarizes clusters by selecting key arguments using method version 11 with spaCy sentence splitting. (per-topic cached)
def method_v11(cluster_by_topic, model, embedder=""):
    nlp = spacy.load(SPA_MODEL_NAME)
    topics = list(cluster_by_topic.keys())
    final_sorted = []

    for topic in topics:
        topic_safe = _safe_topic_name(topic)
        topic_cache_path = os.path.join(CACHE_BASE, f"summary_v11_{topic_safe}.pkl")

        if CACHE_ENABLED and not FORCE_RECOMPUTE and os.path.exists(topic_cache_path):
            try:
                ss = _load(topic_cache_path)
                final_sorted.append(ss)
                print(f"[CACHE] Loaded summary for topic '{topic_safe}' from cache")
                continue
            except Exception as _e:
                print(
                    f"[CACHE] Failed to load summary cache for topic '{topic_safe}': {_e}. Recomputing…"
                )

        summaries = []
        for cluster in cluster_by_topic[topic]:
            arguments = cluster_by_topic[topic][cluster]
            arguments = [str(word) for line in arguments for word in nlp(line).sents]
            arguments = [x for x in arguments if x]
            arg_and_cov = arguments_coverage_all_candidates(arguments, model)
            sum = top_arg_scoring_v1(topic, arg_and_cov)
            summaries.append([sum, len(arguments)])

        # Sort within this topic by length (descending), then keep only the strings
        s = list(summaries)
        s.sort(reverse=True, key=lambda x: x[1])
        ss = [item[0] for item in s]

        if CACHE_ENABLED:
            try:
                _dump(ss, topic_cache_path)
                print(f"[CACHE] Saved summary for topic '{topic_safe}' to cache")
            except Exception as _e:
                print(
                    f"[CACHE] Could not write summary cache for topic '{topic_safe}': {_e}"
                )

        final_sorted.append(ss)

    return final_sorted


# Limits the output summaries to specified lengths per topic.
def limit_output(sum, limits):
    return [sum[i][: limits[i]] for i in range(len(sum))]


# Clusters arguments by topic using embeddings and agglomerative clustering. (cached)
def cluster(input_args_kp_by_topic, embedder, limits=None):
    clusters_cache_path = os.path.join(CACHE_BASE, "clusters.pkl")

    # Try to load clusters from cache first
    if CACHE_ENABLED and not FORCE_RECOMPUTE and os.path.exists(clusters_cache_path):
        try:
            cluster_by_topic = _load(clusters_cache_path)
            # Update METRICS based on cached content for consistency
            try:
                import numpy as _np

                topics = list(cluster_by_topic.keys())
                all_args = 0
                for topic in topics:
                    for cl in cluster_by_topic[topic]:
                        METRICS["cluster_sizes"].append(
                            len(cluster_by_topic[topic][cl])
                        )
                        all_args += len(cluster_by_topic[topic][cl])
                METRICS["total_arguments"] += all_args
            except Exception:
                pass
            print(f"[CACHE] Loaded clusters from {clusters_cache_path}")
            return cluster_by_topic, ""
        except Exception as _e:
            print(
                f"[CACHE] Failed to load clusters cache {clusters_cache_path}: {_e}. Recomputing…"
            )

    cluster_by_topic = {}
    topics = list(input_args_kp_by_topic.keys())

    for topic in tqdm(topics, desc="Clustering topics"):
        arguments = list(set().union(*input_args_kp_by_topic[topic].values()))
        # Summary-mode: accumulate total unique arguments processed
        try:
            METRICS["total_arguments"] += len(arguments)
        except Exception:
            pass
        _t_emb = time.time()
        corpus_embeddings = embedder.encode(arguments)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )

        if limits:
            n_clusters = limits[topics.index(topic)]
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clustering_model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.5
            )

        clustering_model.fit(corpus_embeddings)
        labels = clustering_model.labels_
        try:
            import numpy as _np

            counts = _np.bincount(labels)
            METRICS["cluster_sizes"].extend(counts.tolist())
        except Exception:
            pass

        cluster_by_topic[topic] = defaultdict(list)
        for sent_id, cl_id in enumerate(labels):
            cluster_by_topic[topic][cl_id].append(arguments[sent_id])

    # Save clusters to cache
    if CACHE_ENABLED:
        try:
            _dump(cluster_by_topic, clusters_cache_path)
            print(f"[CACHE] Saved clusters to {clusters_cache_path}")
        except Exception as _e:
            print(f"[CACHE] Could not write clusters cache {clusters_cache_path}: {_e}")

    return cluster_by_topic, ""


# Loads arguments CSV and builds a nested dictionary by topic.
def build_topic_args_dict(path_to_arguments_csv: str):
    df = pd.read_csv(path_to_arguments_csv)
    topic_args = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        topic = row["topic"]
        topic_args[topic]["ALL"].append(row["argument"])
    return topic_args


# Runs the full experiment pipeline: clustering and summarization. (resumable)
def run_experiment_args_only(model, embedder, topic_args_dict, method):
    print(
        f"[RUN] Experiment started with {len(topic_args_dict)} topics. Clustering into up to {NUM_KEYPOINTS} keypoints per topic."
    )
    t0_total = time.time()

    print("=== Clustering arguments ===")
    clusters, _ = cluster(
        topic_args_dict, embedder, limits=[NUM_KEYPOINTS] * len(topic_args_dict)
    )
    print(f"   ↳ Fertig nach {time.time() - t0_total:.1f}s\n")

    print("=== Generating summaries ===")
    t0_sum = time.time()
    summaries = method(clusters, model, embedder)
    print(f"   ↳ Fertig nach {time.time() - t0_sum:.1f}s")

    print(f"=== Gesamtzeit: {(time.time() - t0_total) / 60:.1f} min ===")
    return summaries


print("=== Running experiment with arguments only ===")
topic_args_only = build_topic_args_dict(arguments_file)

summaries = run_experiment_args_only(model, embedder, topic_args_only, method_v11)

for topic_id, topic in enumerate(topic_args_only):
    print(f"\n=== {topic} ===")
    for kp in summaries[topic_id]:
        print("-", kp)


topics = list(topic_args_only.keys())
# Create long-format CSV with columns: key_point_id, topic, key_point
rows = []
for topic_id, topic in enumerate(topics):
    for key_point_id, key_point in enumerate(summaries[topic_id], start=1):
        rows.append(
            {"key_point_id": key_point_id, "topic": topic, "key_point": key_point}
        )
kp_df = pd.DataFrame(rows)
os.makedirs("./output", exist_ok=True)
kp_df.to_csv("./output/khosravani_keypoints_argmatch_3x300.csv", index=False)

# === SUMMARY REPORT (end-of-run) ===
print("\n=== SUMMARY REPORT (end-of-run) ===")
try:
    import numpy as _np

    # Sequence lengths across all pairs
    if METRICS["seq_lengths"]:
        lens = _np.array(METRICS["seq_lengths"])
        over_512 = int((lens > 512).sum())
        print(
            f"[SUMMARY] Token sequence lengths → count={lens.size}, min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}, median={_np.median(lens):.1f}, >512={over_512}"
        )
    # Overall 'quality' (coverage per argument) across all topics
    if METRICS["coverage_values"]:
        v = _np.array(METRICS["coverage_values"], dtype=float)
        print(
            f"[SUMMARY] Argument-quality (coverage) across ALL arguments → count={v.size}, range=[{v.min():.0f}, {v.max():.0f}], mean={v.mean():.2f}, median={_np.median(v):.2f}"
        )
    # Positive-class probabilities (pairwise) aggregated
    if METRICS["pos_probs"]:
        p = _np.array(METRICS["pos_probs"], dtype=float)
        print(
            f"[SUMMARY] Pairwise positive-class probabilities → count={p.size}, min={p.min():.3f}, max={p.max():.3f}, mean={p.mean():.3f}, std={p.std():.3f}"
        )
        # compact histogram (10 bins)
        hist_counts, bin_edges = _np.histogram(p, bins=_np.linspace(0, 1, 11))
        bins_readable = ", ".join(
            [
                f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}:{int(hist_counts[i])}"
                for i in range(len(hist_counts))
            ]
        )
        print(
            f"[SUMMARY] Probabilities histogram (10 bins over [0,1]): {bins_readable}"
        )
    # Labeled pair totals
    print(
        f"[SUMMARY] Labeled pairs total → positive={METRICS['labeled_pos']}, negative={METRICS['labeled_neg']}"
    )
    # Clustering sizes across topics
    if METRICS["cluster_sizes"]:
        cs = _np.array(METRICS["cluster_sizes"], dtype=int)
        print(
            f"[SUMMARY] Clusters → total clusters={cs.size}, size range=[{cs.min()}, {cs.max()}], mean size={cs.mean():.2f}, median size={_np.median(cs):.2f}"
        )
    # Totals
    print(
        f"[SUMMARY] Total unique arguments processed (across topics): {METRICS['total_arguments']}"
    )
    print(f"[SUMMARY] Total prediction vectors produced: {METRICS['pred_vectors']}")
except Exception as _e:
    print(f"[SUMMARY] Could not compute summary statistics: {_e}")
print("=== End of SUMMARY REPORT ===\n")
