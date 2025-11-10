# -*- coding: utf-8 -*-
"""
Combine Alshomary-style preprocessing (sentence splitting, pronoun filtering,
topic-paired quality scoring, length filtering) with the Khosravani KPG method.

Usage (example):
python KPG_pipeline_alsho2khos.py \
  --arguments-csv ../modified_KPG/output/arguments_processed_300.csv \
  --quality-model-path ./models/argument-quality-ibm-reproduced/bert_wa \
  --embedder-path ./models/V1 \
  --coverage-clf-path ./models/2 \
  --num-keypoints 10 \
  --min-quality-score 0.60 \
  --min-len 5 --max-len 20 \
  --output-path ./output/keypoints_alsho2khos.csv
"""

# =========================
# Standard Library
# =========================
import os
import re
import time
import json
import pickle
import argparse
import hashlib
from collections import defaultdict

# =========================
# Third Party
# =========================
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import softmax

# NLP
import spacy
from spacy.lang.en import English

# Torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
)

# Sentence-Transformers
from sentence_transformers import SentenceTransformer

# Clustering
from sklearn.cluster import AgglomerativeClustering


# =========================
# CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alshomary-preprocessed → Khosravani KPG pipeline"
    )
    p.add_argument(
        "--arguments-csv",
        required=True,
        help="CSV with at least columns: topic, argument (stance optional)",
    )
    p.add_argument(
        "--output-path",
        default="./output/keypoints_alsho2khos.csv",
        help="Where to save resulting key points (CSV)",
    )
    p.add_argument(
        "--quality-model-path",
        default="./models/argument-quality-ibm-reproduced/bert_wa",
        help="Path to fine-tuned BERT model for quality scoring (HF or custom Webis)",
    )
    p.add_argument(
        "--embedder-path",
        default="./models/V1",
        help="SentenceTransformer model for clustering (Khosravani)",
    )
    p.add_argument(
        "--coverage-clf-path",
        default="./models/2",
        help="BERT sequence-classification checkpoint used in Khosravani coverage",
    )
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=10,
        help="Number of clusters (keypoints) per topic",
    )
    p.add_argument(
        "--min-quality-score",
        type=float,
        default=0.60,
        help="Min quality prob to keep a sentence (Alshomary preprocessing)",
    )
    p.add_argument(
        "--min-len", type=int, default=5, help="Min sentence length (tokens) to keep"
    )
    p.add_argument(
        "--max-len", type=int, default=20, help="Max sentence length (tokens) to keep"
    )
    p.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA available"
    )
    p.add_argument(
        "--cache-dir", default="./output/cache_alsho2khos", help="Cache directory"
    )
    return p


# =========================
# Helpers (cache etc.)
# =========================
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


# =========================
# Alshomary: Quality model loader
# =========================
class CustomBERTModel(torch.nn.Module):
    """BERT backbone + dropout + linear(768->1) + sigmoid at the end.
    Returns probabilities in [0,1].
    """

    def __init__(self, num_labels: int = 1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        with torch.no_grad():
            self.classifier.bias.fill_(0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS
        x = self.dropout(pooled)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


def load_quality_model(model_name: str, device: torch.device):
    """Load either a custom Webis checkpoint (model.pt) or a HF seq-clf model.
    Returns (tokenizer, model, is_custom).
    """
    is_dir = os.path.isdir(model_name)
    custom_ckpt = is_dir and os.path.isfile(os.path.join(model_name, "model.pt"))

    if custom_ckpt:
        # Read num_labels if available (fallback 1)
        num_labels = 1
        tc_path = os.path.join(model_name, "training_config.json")
        if os.path.isfile(tc_path):
            try:
                with open(tc_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                num_labels = int(cfg.get("num_labels", 1))
            except Exception:
                pass
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = CustomBERTModel(num_labels=num_labels)
        state = torch.load(os.path.join(model_name, "model.pt"), map_location=device)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return tokenizer, model, True

    # HF style model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    return tokenizer, model, False


# =========================
# Devices & Globals
# =========================
DEVICE = None
QUALITY_TOKENIZER = None
QUALITY_MODEL = None
QUALITY_MODEL_IS_CUSTOM = False

# spaCy components for preprocessing
_sent_pipe = English()
_sent_pipe.add_pipe("sentencizer")
try:
    _ = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


# =========================
# Alshomary: preprocessing steps
# =========================
def split_and_filter(text: str) -> list[str]:
    """Split text into sentences and drop those starting with a pronoun."""
    sents = [s.text for s in _sent_pipe(text).sents]
    kept = []
    for s in sents:
        doc = nlp(s)
        if len(doc) == 0:
            continue
        if doc[0].pos_ == "PRON":
            continue
        kept.append(s)
    return kept


@torch.inference_mode()
def score_sentences(
    topic: str, sentences: list[str], max_len: int = 128
) -> list[tuple[str, float]]:
    """Return list of (sentence, quality_probability)."""
    if not sentences:
        return []
    topics = [topic] * len(sentences)
    enc = QUALITY_TOKENIZER(
        sentences,
        topics,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(DEVICE)

    if QUALITY_MODEL_IS_CUSTOM:
        probs = (
            QUALITY_MODEL(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        scores = probs.tolist()
    else:
        logits = QUALITY_MODEL(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()

    return list(zip(sentences, [float(s) for s in scores]))


def alshomary_preprocess(
    args_df: pd.DataFrame, min_quality: float, min_len: int, max_len: int
) -> dict:
    """Return dict: topic -> {'ALL': [preselected_sentences]}"""
    # 1) sentence segmentation + pronoun filter
    args_df = args_df.copy()
    args_df["sents"] = args_df["argument"].apply(split_and_filter)

    # 2) quality scores per sentence (paired with topic)
    args_df["sents_with_scores"] = args_df.apply(
        lambda r: score_sentences(r["topic"], r["sents"]), axis=1
    )

    # 3) keep by thresholds (quality + length)
    topic_to_sents = defaultdict(list)
    for _, r in args_df.iterrows():
        topic = r["topic"]
        for sent, q in r["sents_with_scores"]:
            tok_len = len(sent.split())
            if q >= min_quality and min_len <= tok_len <= max_len:
                topic_to_sents[topic].append(sent)

    # 4) build nested dict as Khosravani expects
    topic_args = defaultdict(lambda: defaultdict(list))
    for t, sents in topic_to_sents.items():
        # de-duplicate while keeping order
        seen = set()
        uniq = []
        for s in sents:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        topic_args[t]["ALL"] = uniq
    return topic_args


# =========================
# Khosravani: coverage + clustering stack
# =========================
def get_tokenizer_bert_base_uncased():
    return BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def get_optimizer_for(model):
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
    return AdamW(optimizer_grouped_parameters, lr=2e-5)


def bert_pair_ds(pairs, tokenizer):
    token_ids, mask_ids, seg_ids = [], [], []
    lengths = []
    for prem, hyp in pairs:
        premise_id = tokenizer.encode(prem, add_special_tokens=False)
        hypothesis_id = tokenizer.encode(hyp, add_special_tokens=False)
        pair_token_ids = (
            [tokenizer.cls_token_id]
            + premise_id
            + [tokenizer.sep_token_id]
            + hypothesis_id
            + [tokenizer.sep_token_id]
        )
        lengths.append(len(pair_token_ids))
        premise_len = len(premise_id)
        hypothesis_len = len(hypothesis_id)
        segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))
        attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))
        token_ids.append(torch.tensor(pair_token_ids))
        seg_ids.append(segment_ids)
        mask_ids.append(attention_mask_ids)

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    return TensorDataset(token_ids, mask_ids, seg_ids)


@torch.inference_mode()
def get_preds(loader, model, device):
    optimizer = get_optimizer_for(model)  # zero_grad per batch (kept from original)
    preds = []
    for pair_token_ids, mask_ids, seg_ids in loader:
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        out = model(pair_token_ids, mask_ids, seg_ids)
        preds.extend(out[0].tolist())
    return preds


def arguments_coverage_all_candidates(arguments_list, model, device):
    """Compute coverage via pairwise BERT classifier → count of 'positive' labels."""
    cache_key = _hash_list(arguments_list) + f"_n{len(arguments_list)}"
    cov_cache_path = os.path.join(CACHE_BASE, f"cov_{cache_key}.pkl")
    if CACHE_ENABLED and os.path.exists(cov_cache_path):
        try:
            return _load(cov_cache_path)
        except Exception:
            pass

    tokenizer = get_tokenizer_bert_base_uncased()
    arg_pairs = [[a, b] for a in arguments_list for b in arguments_list]
    dsa = bert_pair_ds(arg_pairs, tokenizer)
    loader = DataLoader(dsa, shuffle=False, batch_size=128)
    preds = get_preds(loader, model, device)
    soft_preds = softmax(preds, axis=1)

    labeled = [(1 if p[1] > p[0] and p[1] > 0 else 0) for p in preds]
    arg_coverage = {}
    n = len(arguments_list)
    for i in range(n):
        s1 = labeled[i * n : (i + 1) * n]
        arg_coverage[arguments_list[i]] = int(sum(s1))

    if CACHE_ENABLED:
        try:
            _dump(arg_coverage, cov_cache_path)
        except Exception:
            pass
    return arg_coverage


def top_arg_scoring_v1(arg_cov: dict) -> str:
    best = None
    best_score = -1.0
    for arg, cov in arg_cov.items():
        toks = len(arg.split()) if arg else 0
        if toks <= 0:
            continue
        score = (cov**5) / toks
        if score > best_score or (
            score == best_score and arg and best and len(arg) < len(best)
        ):
            best = arg
            best_score = score
    return best


def method_v11(cluster_by_topic, model, device):
    """Khosravani’s method_v11, assuming items are already sentences."""
    final_sorted = []
    topics = list(cluster_by_topic.keys())
    for topic in topics:
        topic_safe = _safe_topic_name(topic)
        topic_cache_path = os.path.join(CACHE_BASE, f"summary_v11_{topic_safe}.pkl")
        if CACHE_ENABLED and os.path.exists(topic_cache_path):
            try:
                ss = _load(topic_cache_path)
                final_sorted.append(ss)
                print(f"[CACHE] Loaded summary for topic '{topic_safe}' from cache")
                continue
            except Exception:
                pass

        summaries = []
        for cluster in cluster_by_topic[topic]:
            arguments = cluster_by_topic[topic][cluster]
            # Already sentence-level from preprocessing; just drop empties
            arguments = [x for x in arguments if x]
            arg_and_cov = arguments_coverage_all_candidates(arguments, model, device)
            chosen = top_arg_scoring_v1(arg_and_cov)
            summaries.append([chosen, len(arguments)])

        s = sorted(summaries, key=lambda x: x[1], reverse=True)
        ss = [item[0] for item in s]

        if CACHE_ENABLED:
            try:
                _dump(ss, topic_cache_path)
                print(f"[CACHE] Saved summary for topic '{topic_safe}' to cache")
            except Exception:
                pass

        final_sorted.append(ss)
    return final_sorted


def cluster(input_args_by_topic, embedder, limits=None):
    """Agglomerative clustering per topic using sentence embeddings."""
    clusters_cache_path = os.path.join(CACHE_BASE, "clusters.pkl")
    if CACHE_ENABLED and os.path.exists(clusters_cache_path):
        try:
            cluster_by_topic = _load(clusters_cache_path)
            print(f"[CACHE] Loaded clusters from {clusters_cache_path}")
            return cluster_by_topic, ""
        except Exception:
            pass

    cluster_by_topic = {}
    topics = list(input_args_by_topic.keys())
    for topic in tqdm(topics, desc="Clustering topics"):
        arguments = list(set().union(*input_args_by_topic[topic].values()))
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

        cluster_by_topic[topic] = defaultdict(list)
        for sent_id, cl_id in enumerate(labels):
            cluster_by_topic[topic][int(cl_id)].append(arguments[sent_id])

    if CACHE_ENABLED:
        try:
            _dump(cluster_by_topic, clusters_cache_path)
            print(f"[CACHE] Saved clusters to {clusters_cache_path}")
        except Exception:
            pass
    return cluster_by_topic, ""


# =========================
# Globals for run
# =========================
CACHE_ENABLED = True
FORCE_RECOMPUTE = False
CACHE_BASE = None


def run_pipeline(args):
    # Device
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print("#" * 35)
    print("USING CUDA" if use_cuda else "USING CPU")
    print("#" * 35)

    # Cache dir
    global CACHE_BASE
    os.makedirs(args.cache_dir, exist_ok=True)
    run_hash = hashlib.sha1(
        f"{os.path.abspath(args.arguments_csv)}|{args.embedder_path}|{args.coverage_clf_path}|{args.quality_model_path}|k={args.num_keypoints}|thr={args.min_quality_score}|len={args.min_len}-{args.max_len}".encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    CACHE_BASE = os.path.join(args.cache_dir, run_hash)
    os.makedirs(CACHE_BASE, exist_ok=True)
    print(f"[CACHE] {CACHE_BASE}")

    # Load models
    print("[LOAD] Quality model …")
    global QUALITY_TOKENIZER, QUALITY_MODEL, QUALITY_MODEL_IS_CUSTOM
    QUALITY_TOKENIZER, QUALITY_MODEL, QUALITY_MODEL_IS_CUSTOM = load_quality_model(
        args.quality_model_path, device
    )

    print(f"[LOAD] Sentence embedder from '{args.embedder_path}' …")
    embedder = SentenceTransformer(args.embedder_path)

    print(f"[LOAD] Coverage classifier from '{args.coverage_clf_path}' …")
    cov_model = (
        BertForSequenceClassification.from_pretrained(
            args.coverage_clf_path, num_labels=2
        )
        .to(device)
        .eval()
    )

    # Data
    df = pd.read_csv(args.arguments_csv)
    if "topic" not in df.columns or "argument" not in df.columns:
        raise ValueError("Input CSV must contain columns: topic, argument")

    # Phase 1: Alshomary-style preprocessing -> topic -> {'ALL': [filtered sentences]}
    print("=== Phase 1: Alshomary preprocessing ===")
    t0 = time.time()
    topic_args = alshomary_preprocess(
        df,
        min_quality=args.min_quality_score,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    print(
        f"   ↳ kept topics: {len(topic_args)}; total kept sentences: {sum(len(v['ALL']) for v in topic_args.values())}"
    )
    print(f"   ↳ done in {time.time() - t0:.1f}s\n")

    # Phase 2: Khosravani clustering + coverage summarization
    print("=== Phase 2: Khosravani clustering & summarization ===")
    t1 = time.time()
    clusters, _ = cluster(
        topic_args, embedder, limits=[args.num_keypoints] * len(topic_args)
    )
    summaries = method_v11(clusters, cov_model, device)
    print(f"   ↳ done in {time.time() - t1:.1f}s")

    # Save long-format CSV
    topics = list(topic_args.keys())
    rows = []
    for topic_id, topic in enumerate(topics):
        for key_point_id, key_point in enumerate(summaries[topic_id], start=1):
            rows.append(
                {"key_point_id": key_point_id, "topic": topic, "key_point": key_point}
            )
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"[OUTPUT] Saved key-points to: {args.output_path}")

    # Short summary
    print("\n=== SUMMARY ===")
    n_kept = sum(len(v["ALL"]) for v in topic_args.values())
    print(f"[DATA] kept sentences after preprocessing: {n_kept}")
    print(
        f"[KPG] topics: {len(topics)}, keypoints per topic (target): {args.num_keypoints}"
    )
    print(f"[KPG] total keypoints written: {len(out_df)}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

"""

python src/codereplication/khosravani2024/KPG_pipeline_alsho2khos.py \
  --arguments-csv ../modified_KPG/output/arguments_processed_300.csv \
  --quality-model-path ./models/argument-quality-ibm-reproduced/bert_wa \
  --embedder-path ./models/V1 \
  --coverage-clf-path ./models/2 \
  --num-keypoints 10 \
  --min-quality-score 0.60 \
  --min-len 5 --max-len 20 \
  --output-path ./output/keypoints_alsho2khos.csv

"""
