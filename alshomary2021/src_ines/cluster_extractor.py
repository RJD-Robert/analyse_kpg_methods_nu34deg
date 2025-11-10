#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 1:
- Read arguments CSV
- Sentence segmentation & filtering
- Quality scoring with your fine-tuned BERT
- Aggregate into (topic, stance) "clusters"
- Write clusters to JSONL

Output schema (JSONL, one object per line):
{
  "topic": "string",
  "stance": "string",
  "candidates": [
      {"sent": "sentence text", "score": float},
      ...
  ]
}
"""

import argparse
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification


# ----------------------------- CLI ----------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True,
                   help="Path to arguments_combined.csv")
    p.add_argument("--output_clusters", default="clusters.jsonl",
                   help="Where to write the clusters JSONL")
    p.add_argument("--model_path", required=True,
                   help="Path to your fine-tuned BERT (with model.pt inside)")
    p.add_argument("--base_config", default="bert-base-uncased",
                   help="Base BERT config name (must be available locally if offline)")
    p.add_argument("--max_len", type=int, default=128,
                   help="Max sequence length for the quality model")
    return p.parse_args()


# ----------------------------- NLP setup ----------------------------------- #
sent_pipe = English()
sent_pipe.add_pipe("sentencizer")

# Full spaCy model for POS tags (pronoun filter)
nlp = spacy.load("en_core_web_sm")

def split_and_filter(text: str) -> List[str]:
    """
    1) Split into sentences.
    2) Drop sentences that start with a pronoun.
    """
    out = []
    for sent in sent_pipe(text).sents:
        doc = nlp(sent.text)
        if not doc:  # empty / weird
            continue
        if doc[0].pos_ != "PRON":
            out.append(sent.text)
    return out


# -------------------------- Quality model ---------------------------------- #
def load_quality_model(model_path: str, base_config: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config    = BertConfig.from_pretrained(base_config, num_labels=1)
    model     = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def score_sentences(
    topic: str,
    sents: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_len: int
) -> List[Tuple[str, float]]:
    """
    Returns [(sentence, score)], where higher score = better.
    """
    if not sents:
        return []

    enc = tokenizer(
        sents,
        [topic] * len(sents),
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    logits = model(**enc).logits.squeeze(-1)
    scores = (-logits).tolist()  # invert: lower logit => higher quality
    return list(zip(sents, scores))


def merge_candidates(pairs: List[Tuple[str, float]]) -> List[Dict]:
    """
    Deduplicate sentences; keep the highest score per sentence.
    Returns a list of dicts ready to be JSON-serialized.
    """
    best = {}
    for sent, score in pairs:
        if sent not in best or score > best[sent]:
            best[sent] = score
    return [{"sent": s, "score": float(sc)} for s, sc in best.items()]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[preprocess_to_clusters] Using device: {device}")

    # Load model
    tokenizer, q_model = load_quality_model(args.model_path, args.base_config, device)

    # Read data
    df = pd.read_csv(args.input_csv)

    # 1) sentence segmentation & filtering
    print("[preprocess_to_clusters] Splitting & filtering sentences...")
    df["sents"] = df["argument"].apply(split_and_filter)

    # 2) quality scoring
    print("[preprocess_to_clusters] Scoring sentences...")
    df["sents_with_scores"] = df.apply(
        lambda r: score_sentences(
            r["topic"], r["sents"], tokenizer, q_model, device, args.max_len
        ),
        axis=1
    )

    # 3) cluster by (topic, stance)
    print("[preprocess_to_clusters] Building clusters...")
    clusters_df = (
        df.groupby(["topic", "stance"])["sents_with_scores"]
          .apply(lambda col: merge_candidates([p for pairs in col for p in pairs]))
          .reset_index()
          .rename(columns={"sents_with_scores": "candidates"})
    )

    # 4) write JSONL
    print(f"[preprocess_to_clusters] Writing clusters to {args.output_clusters} ...")
    with open(args.output_clusters, "w", encoding="utf-8") as f:
        for _, row in clusters_df.iterrows():
            obj = {
                "topic": row["topic"],
                "stance": row["stance"],
                "candidates": row["candidates"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[preprocess_to_clusters] Done. Wrote {len(clusters_df)} clusters.")


if __name__ == "__main__":
    main()