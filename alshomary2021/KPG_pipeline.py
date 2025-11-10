import argparse
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
import torch

from fast_pagerank import pagerank
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertModel,
)

# =============================================================================
# Globals that are initialized in main() after parsing CLI arguments
# =============================================================================

# Models (initialized in main)
tokenizer = None  # type: ignore
quality_model = None  # type: ignore
sim_model = None  # type: ignore
QUALITY_MODEL_IS_CUSTOM = (
    False  # whether the quality model already outputs probabilities
)

# Device (initialized in main)
DEVICE = None  # type: ignore

# Hyperparameters (can be overridden via CLI)
PAGERANK_P = 0.80
MIN_QUALITY_SCORE = 0.60
MIN_MATCH_SCORE_PR = 0.50
MIN_MATCH_FILTER = 0.75
MIN_LEN = 5
MAX_LEN = 20
TOP_K = 5

# =============================================================================
# NLP Components
# =============================================================================

# Lightweight sentence splitter (no full pipeline)
sent_pipe = English()
sent_pipe.add_pipe("sentencizer")

# Full spaCy model for POS tagging (to filter out pronouns)
nlp = spacy.load("en_core_web_sm")


# =============================================================================
# Global Matching-Score Accumulators (for end-of-run summary only)
# =============================================================================

SIM_STATS = {
    "raw": {
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "bins": [0] * 20,
    },
    "retained": {
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "bins": [0] * 10,
    },
}


# =============================================================================
# Quality Model Loader (matches usage in arg_quality_csv.py)
# =============================================================================
class CustomBERTModel(torch.nn.Module):
    """BERT backbone + dropout + linear(768->1) + sigmoid at the end.
    The forward() returns probabilities in [0,1].
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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_quality_model(model_name: str, device: torch.device):
    """Load either a custom Webis checkpoint (model.pt) or a HF seq-clf model.
    Returns (tokenizer, model, is_custom).
    """
    is_dir = os.path.isdir(model_name)
    custom_ckpt = is_dir and os.path.isfile(os.path.join(model_name, "model.pt"))

    if custom_ckpt:
        # Read num_labels if available
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

    # HF style model (expects a sequence-classification checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    return tokenizer, model, False


# =============================================================================
# Utility Functions
# =============================================================================


def split_and_filter(text: str) -> list[str]:
    """Split *text* into sentences and drop those starting with a pronoun.

    We use a lightweight sentencizer to split, and the full spaCy model to
    check the POS tag of the first token.
    """
    return [
        sent.text for sent in sent_pipe(text).sents if nlp(sent.text)[0].pos_ != "PRON"
    ]


@torch.inference_mode()
def score_sentences(
    topic: str, sentences: list[str], max_len: int = 128
) -> list[tuple[str, float]]:
    """Return a list of ``(sentence, quality_score)`` pairs.

    Notes
    -----
    * Encode the sentence **paired with the topic** (as in arg_quality_csv.py).
    * If the quality model is custom, it already returns probabilities in [0,1].
      Otherwise, apply sigmoid(logits) to obtain probabilities.
    """
    if not sentences:
        return []

    topics = [topic] * len(sentences)
    enc = tokenizer(
        sentences,
        topics,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(DEVICE)

    if QUALITY_MODEL_IS_CUSTOM:
        probs = (
            quality_model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        scores = probs.tolist()
    else:
        logits = quality_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()  # probabilities in [0,1]

    return list(zip(sentences, [float(s) for s in scores]))


def _accumulate_raw_similarities(mat: np.ndarray) -> None:
    """Accumulate statistics for all off-diagonal cosine similarities."""
    n = mat.shape[0]
    if n <= 1:
        return

    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    if vals.size == 0:
        return

    stats = SIM_STATS["raw"]
    stats["count"] += int(vals.size)
    stats["sum"] += float(vals.sum())

    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin < stats["min"]:
        stats["min"] = vmin
    if vmax > stats["max"]:
        stats["max"] = vmax

    # Map [-1, 1] to 20 bins [0..19]
    idxs = (((vals + 1.0) / 2.0) * 20).astype(int)
    idxs[idxs < 0] = 0
    idxs[idxs > 19] = 19
    for i in idxs:
        stats["bins"][int(i)] += 1


def _accumulate_retained_similarities(mat: np.ndarray) -> None:
    """Accumulate statistics for retained (thresholded) edges only."""
    n = mat.shape[0]
    if n <= 1:
        return

    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    vals = vals[vals > 0.0]  # only edges that survived thresholding
    if vals.size == 0:
        return

    stats = SIM_STATS["retained"]
    stats["count"] += int(vals.size)
    stats["sum"] += float(vals.sum())

    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin < stats["min"]:
        stats["min"] = vmin
    if vmax > stats["max"]:
        stats["max"] = vmax

    # Map [0, 1] to 10 bins [0..9]
    clipped = np.clip(vals, 0.0, 0.999999)
    idxs = (clipped * 10).astype(int)
    for i in idxs:
        stats["bins"][int(i)] += 1


def cosine_graph(cands: list[str], topic: str, min_thr: float) -> np.ndarray:
    """Build a cosine-similarity graph for *cands* (topic not used).

    Steps
    -----
    1. Encode each candidate sentence (topic-independent).
    2. Compute cosine-similarity matrix (matching score proxy).
    3. Zero out edges below ``min_thr``.
    """
    emb = sim_model.encode(cands)
    mat = cosine_similarity(emb)

    _accumulate_raw_similarities(mat)
    mat[mat < min_thr] = 0.0
    _accumulate_retained_similarities(mat)

    return mat


def rank_sentences(candidates: list[tuple[str, float]], topic: str) -> list[str]:
    """Rank sentences using Personalized PageRank seeded by quality scores."""
    if not candidates:
        return []

    sents, qual = zip(*candidates)

    # Similarity graph
    graph = cosine_graph(list(sents), topic, MIN_MATCH_SCORE_PR)

    # Normalize quality vector (fallback to uniform if degenerate)
    qual_vec = np.array(qual, dtype=float)
    if qual_vec.sum() > 0:
        qual_vec = qual_vec / qual_vec.sum()
    else:
        qual_vec = np.full_like(qual_vec, 1.0 / len(qual_vec))

    pr_scores = pagerank(graph, personalize=qual_vec, p=PAGERANK_P)

    # Sort descending by PageRank score
    ranked = sorted(zip(sents, pr_scores), key=lambda x: -x[1])
    return [s for s, _ in ranked]


def diversify(
    ranked: list[str],
    topic: str,
    min_match: float = None,
    top_k: int = None,
) -> list[str]:
    """Apply diversity filtering as in the paper.

    Iterate ranked sentences; keep a sentence only if its max matching score
    (same signal as graph; here cosine similarity) to already kept sentences is
    ``< min_match``.
    """
    if min_match is None:
        min_match = MIN_MATCH_FILTER
    if top_k is None:
        top_k = TOP_K

    if not ranked:
        return []

    kept: list[str] = []
    for sent in ranked:
        if not kept:
            kept.append(sent)
        else:
            sims = cosine_graph([sent] + kept, topic, 0.0)[0, 1:]
            if sims.max() < min_match:
                kept.append(sent)
        if len(kept) >= top_k:
            break
    return kept


def preprocess(args_path: str) -> pd.DataFrame:
    """Phase 1: load CSV, split/filter sentences, and score sentence quality."""
    args_df = pd.read_csv(args_path)

    # Sentence segmentation + pronoun filter
    args_df["sents"] = args_df["argument"].apply(split_and_filter)

    # Quality scoring per sentence
    args_df["sents_with_scores"] = args_df.apply(
        lambda r: score_sentences(r["topic"], r["sents"]), axis=1
    )

    return args_df


def generate_keypoints(args_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: cluster by (topic, stance), rank, diversify, and flatten."""

    # ── Cluster creation: merge list-of-lists into a single set of (sent, score)
    grouped = (
        args_df.groupby(["topic", "stance"]).agg(
            candidates=(
                "sents_with_scores",
                lambda col: {pair for pairs in col for pair in pairs},
            )
        )
    ).reset_index()

    # Rank sentences within each cluster
    grouped["ranked"] = grouped.apply(
        lambda r: rank_sentences(
            [
                p
                for p in r["candidates"]
                if p[1] > MIN_QUALITY_SCORE and MIN_LEN < len(p[0].split()) < MAX_LEN
            ],
            r["topic"],
        ),
        axis=1,
    )

    # Diversity filtering within each cluster
    grouped["key_points"] = grouped.apply(
        lambda r: diversify(r["ranked"], r["topic"]), axis=1
    )

    # Flatten to DataFrame: one row per key-point
    rows: list[list[object]] = []
    kp_id = 0
    for _, r in grouped.iterrows():
        for kp in r["key_points"]:
            rows.append([kp_id, r["topic"], r["stance"], kp])
            kp_id += 1

    return pd.DataFrame(rows, columns=["key_point_id", "topic", "stance", "key_point"])


# =============================================================================
# CLI & main()
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Key Point Generation pipeline (Alshomary 2021) with CLI parameters."
    )
    parser.add_argument(
        "--model-path",
        default="./models/argument-quality-ibm-reproduced/bert_wa",
        help="Path to fine-tuned BERT model for quality scoring",
    )
    parser.add_argument(
        "--sim-model-path",
        default="./models/roberta-large-final-model-fold-4-2023-07-05_16-02-50",
        help="Path to SentenceTransformer model used for similarity",
    )
    parser.add_argument(
        "--arguments-csv",
        default="../data/ibm/ArgKP_arguments_3_topics.csv",
        help="CSV of test arguments",
    )
    parser.add_argument(
        "--output-path",
        default="./output/alshomary_keypoints_Ines_3_topics.csv",
        help="Where to save the resulting key-points CSV",
    )
    parser.add_argument(
        "--pagerank-p", type=float, default=0.80, help="PageRank p (follow-edge prob)"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.75,
        help="Minimum quality score to keep a sentence",
    )
    parser.add_argument(
        "--min-match-score-pr",
        type=float,
        default=0.50,
        help="Edge exists if cosine similarity ≥ this for PageRank graph",
    )
    parser.add_argument(
        "--min-match-filter",
        type=float,
        default=0.75,
        help="Redundancy filter threshold for diversity step",
    )
    parser.add_argument(
        "--min-len", type=int, default=5, help="Minimum sentence length (tokens)"
    )
    parser.add_argument(
        "--max-len", type=int, default=20, help="Maximum sentence length (tokens)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of key points to keep per (topic, stance)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    return parser


def main():
    global tokenizer, quality_model, sim_model
    global DEVICE
    global \
        PAGERANK_P, \
        MIN_QUALITY_SCORE, \
        MIN_MATCH_SCORE_PR, \
        MIN_MATCH_FILTER, \
        MIN_LEN, \
        MAX_LEN, \
        TOP_K, \
        QUALITY_MODEL_IS_CUSTOM

    parser = build_arg_parser()
    args = parser.parse_args()

    # Set device
    use_cuda = torch.cuda.is_available() and not args.cpu
    print("#" * 35)
    print("USING CUDA" if use_cuda else "USING CPU")
    print("#" * 35)
    DEVICE = torch.device("cuda" if use_cuda else "cpu")

    # Override hyperparameters from CLI
    PAGERANK_P = float(args.pagerank_p)
    MIN_QUALITY_SCORE = float(args.min_quality_score)
    MIN_MATCH_SCORE_PR = float(args.min_match_score_pr)
    MIN_MATCH_FILTER = float(args.min_match_filter)
    MIN_LEN = int(args.min_len)
    MAX_LEN = int(args.max_len)
    TOP_K = int(args.top_k)

    # Load quality model + tokenizer (supports custom Webis or HF checkpoints)
    tokenizer, quality_model, QUALITY_MODEL_IS_CUSTOM = load_quality_model(
        args.model_path, DEVICE
    )

    # Sentence-BERT model for similarity
    sim_model = SentenceTransformer(args.sim_model_path)

    # Run pipeline
    test_args = preprocess(args.arguments_csv)
    keypoints_df = generate_keypoints(test_args)

    # Save output
    keypoints_df.to_csv(args.output_path, index=False)

    # =============== END-OF-RUN SUMMARY (compact) ===============
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    # Dataset
    print(f"[DATA] Arguments: {len(test_args)}")

    # Sentences per argument
    if "sents" in test_args.columns:
        _sent_counts = test_args["sents"].apply(len)
        if len(_sent_counts) > 0:
            print(
                "[SENTS] Sentences per argument → min={}, mean={:.2f}, median={:.2f}, max={}".format(
                    _sent_counts.min(),
                    _sent_counts.mean(),
                    _sent_counts.median(),
                    _sent_counts.max(),
                )
            )

    # Global quality score distribution across ALL sentences
    _all_scores = []
    if "sents_with_scores" in test_args.columns:
        for _pairs in test_args["sents_with_scores"]:
            for _sent, _sc in _pairs:
                _all_scores.append(_sc)
    if _all_scores:
        _all_scores_sorted = sorted(_all_scores)
        _n = len(_all_scores_sorted)

        def _q(p: float) -> float:
            _idx = min(max(int(p * _n), 0), _n - 1)
            return _all_scores_sorted[_idx]

        print(
            "[QUALITY] Scores across ALL sentences → count={}, min={:.4f}, p25={:.4f}, median={:.4f}, p75={:.4f}, max={:.4f}, mean={:.4f}".format(
                _n,
                _all_scores_sorted[0],
                _q(0.25),
                _q(0.5),
                _q(0.75),
                _all_scores_sorted[-1],
                sum(_all_scores_sorted) / _n,
            )
        )
        _bins = [0] * 10
        for _s in _all_scores_sorted:
            _idx = 9 if _s >= 0.999999 else int(_s * 10)
            _bins[_idx] += 1
        print(
            "[QUALITY] Histogram (0.0–0.1 | … | 0.9–1.0): "
            + ", ".join(str(b) for b in _bins)
        )
    else:
        print(
            "[QUALITY] No quality scores computed (no sentences after filtering or empty input)."
        )

    # Matching scores (cosine similarity) overview across ALL graphs
    if SIM_STATS["raw"]["count"] > 0:
        raw = SIM_STATS["raw"]
        raw_mean = raw["sum"] / raw["count"]
        print(
            "[MATCH|RAW] Cosine similarities before threshold → count={}, min={:.4f}, mean={:.4f}, max={:.4f}".format(
                raw["count"], raw["min"], raw_mean, raw["max"]
            )
        )
        print(
            "[MATCH|RAW] Histogram 20 bins (-1.0–-0.9 | … | 0.9–1.0): "
            + ", ".join(str(b) for b in raw["bins"])
        )
    else:
        print("[MATCH|RAW] No raw pairwise similarities recorded.")

    if SIM_STATS["retained"]["count"] > 0:
        kept = SIM_STATS["retained"]
        kept_mean = kept["sum"] / kept["count"]
        print(
            "[MATCH|RETAINED] Similarities of retained edges (≥ {:.2f}) → count={}, min={:.4f}, mean={:.4f}, max={:.4f}".format(
                MIN_MATCH_SCORE_PR, kept["count"], kept["min"], kept_mean, kept["max"]
            )
        )
        print(
            "[MATCH|RETAINED] Histogram 10 bins (0.0–0.1 | … | 0.9–1.0): "
            + ", ".join(str(b) for b in kept["bins"])
        )
    else:
        print("[MATCH|RETAINED] No retained edges (threshold too high or empty input).")

    # Key-points
    print(f"[KPG] Total key-points: {len(keypoints_df)}")

    # Output
    print(f"[OUTPUT] Saved key-points to: {args.output_path}")


if __name__ == "__main__":
    main()
