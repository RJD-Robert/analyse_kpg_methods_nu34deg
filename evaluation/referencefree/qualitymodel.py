#!/usr/bin/env python3
# save as: arg_quality_csv.py
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
from typing import Optional, List
import os
import json


class CustomBERTModel(torch.nn.Module):
    """Matches the architecture used in the Webis repo when model_type=='custom'.
    BERT backbone + dropout + linear(768->1) + sigmoid at the end.
    The forward() returns probabilities in [0,1].
    """

    def __init__(self, num_labels: int = 1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        # Match Webis init behavior
        with torch.no_grad():
            self.classifier.bias.fill_(0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model(model_name: str, device: Optional[str] = None):
    """Load either a HF sequence-classification checkpoint or the Webis custom checkpoint.
    If the directory contains `model.pt` (and `training_config.json`), we assume the
    Webis custom architecture and load `CustomBERTModel` + `BertTokenizer`.
    Otherwise, fall back to AutoModelForSequenceClassification.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect custom checkpoint (like your local bert_wa folder)
    is_dir = os.path.isdir(model_name)
    custom_ckpt = is_dir and os.path.isfile(os.path.join(model_name, "model.pt"))

    if custom_ckpt:
        print(f"Loading custom checkpoint from {model_name} on device {device}")
        # Read num_labels if available (defaults to 1)
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
        return tokenizer, model, device, True  # True -> custom model (already sigmoid)

    # HF style model (expects pytorch_model.bin / safetensors)
    print(f"Loading HF model from {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    return tokenizer, model, device, False


def batched(iterable: List, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def score_csv(
    csv_path: str,
    arg_col: str,
    topic_col: Optional[str],
    out_path: str,
    model_name: str = "../../Alshomary2021/models/argument-quality-ibm-reproduced/bert_wa",
    default_topic: Optional[str] = None,
    batch_size: int = 16,
):
    df = pd.read_csv(csv_path)
    # Texte vorbereiten
    # Wir deduplizieren (Argument, Topic)-Paare für schnellere Berechnung,
    # mappen die Scores danach zurück auf die Originalzeilen.
    if arg_col not in df.columns:
        raise ValueError(f"Spalte '{arg_col}' nicht in CSV gefunden.")
    if topic_col is None and default_topic is None:
        raise ValueError(
            "Entweder '--topic-col' angeben oder '--default-topic' setzen."
        )

    # Erzeuge Arbeits-DataFrame mit expliziten Spaltennamen
    if topic_col is not None:
        if topic_col not in df.columns:
            raise ValueError(f"Spalte '{topic_col}' nicht in CSV gefunden.")
        df_pairs = pd.DataFrame(
            {
                "___arg": df[arg_col].fillna("").astype(str),
                "___topic": df[topic_col].fillna("").astype(str),
            }
        )
    else:
        df_pairs = pd.DataFrame(
            {
                "___arg": df[arg_col].fillna("").astype(str),
                "___topic": [default_topic] * len(df),
            }
        )

    total_rows = len(df_pairs)
    print(f"Eingelesen: {total_rows} Zeilen.")

    # Deduplizieren nach (Argument, Topic)
    df_unique = df_pairs.drop_duplicates(subset=["___arg", "___topic"]).reset_index(
        drop=True
    )
    print(
        f"Eindeutige (Argument,Topic)-Paare: {len(df_unique)} (Einsparung: {total_rows - len(df_unique)})"
    )

    # Listen für das Scoring
    u_args = df_unique["___arg"].tolist()
    u_topics = df_unique["___topic"].tolist()

    tokenizer, model, device, is_custom = load_model(model_name)

    print(f"Batch size: {batch_size}")
    all_scores: List[float] = []
    with torch.no_grad():
        for batch_idx, (arg_batch, topic_batch) in enumerate(
            zip(batched(u_args, batch_size), batched(u_topics, batch_size))
        ):
            print(f"Verarbeite Batch {batch_idx + 1} mit {len(arg_batch)} Einträgen...")
            enc = tokenizer(
                arg_batch,
                topic_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            if is_custom:
                # Custom model already returns probabilities in [0,1]
                probs = (
                    model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            else:
                logits = model(**enc).logits.squeeze(-1).detach().cpu().numpy()
                probs = sigmoid(logits).astype(float)

            all_scores.extend(probs.tolist())

    # Mappe Scores zurück auf die Originalzeilen über (Argument, Topic)
    df_unique_scores = df_unique.copy()
    df_unique_scores["quality_score"] = all_scores
    df_scored = df_pairs.merge(
        df_unique_scores,
        on=["___arg", "___topic"],
        how="left",
        validate="many_to_one",
    )
    mapped_scores = df_scored["quality_score"].tolist()

    print(f"Erste 5 Scores (unique): {all_scores[:5]}")

    # Compute and print summary statistics
    scores_np = np.array(mapped_scores)
    min_score = np.min(scores_np)
    max_score = np.max(scores_np)
    mean_score = np.mean(scores_np)
    std_score = np.std(scores_np)
    print("=" * 40)
    print("Quality Score Summary Statistics:")
    print(f"  Min:  {min_score:.4f}")
    print(f"  Max:  {max_score:.4f}")
    print(f"  Mean: {mean_score:.4f}")
    print(f"  Std:  {std_score:.4f}")
    print("=" * 40)

    df_out = df.copy()
    df_out["quality_score"] = mapped_scores
    df_out.to_csv(out_path, index=False)
    print(f"Fertig. Geschrieben nach: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Compute Argument Quality Scores (bert_wa) for a CSV."
    )
    ap.add_argument("--csv", required=True, help="Pfad zur Eingabe-CSV")
    ap.add_argument("--arg-col", required=True, help="Spaltenname für Argumente")
    ap.add_argument(
        "--topic-col", required=False, help="Spaltenname für Topics (optional)"
    )
    ap.add_argument(
        "--default-topic",
        required=False,
        help="Ein Topic für alle Zeilen, falls --topic-col fehlt",
    )
    ap.add_argument("--out", required=True, help="Pfad zur Ausgabe-CSV")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--model-name",
        default="../../Alshomary2021/models/argument-quality-ibm-reproduced/bert_wa",
    )
    args = ap.parse_args()

    score_csv(
        csv_path=args.csv,
        arg_col=args.arg_col,
        topic_col=args.topic_col,
        out_path=args.out,
        model_name=args.model_name,
        default_topic=args.default_topic,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
