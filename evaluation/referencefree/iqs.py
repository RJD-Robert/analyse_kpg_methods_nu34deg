#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import os
import json

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        BertTokenizer,
        BertModel,
    )
    import torch
except ImportError as e:
    raise SystemExit(
        "Bitte installiere 'transformers' und 'torch':\n"
        "  pip install transformers torch"
    ) from e


# ---------------------------------------------------------------------------
# IQS: Kern-Helfer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CustomBERTModel and model loading helper
# ---------------------------------------------------------------------------
class CustomBERTModel(torch.nn.Module):
    """Custom model used in Webis repo when a local directory contains a custom checkpoint.
    Architecture: BERT backbone + dropout + linear(768->1) + sigmoid.
    Forward returns probabilities in [0,1].
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
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


def load_model(model_name: str, device: str | None = None):
    """Load either a HF sequence-classification checkpoint or the Webis custom checkpoint.
    If `model_name` is a directory containing `model.pt`, we assume the custom architecture
    and load `CustomBERTModel` plus `BertTokenizer`. Otherwise fall back to HF auto classes.
    Returns (tokenizer, model, device, is_custom).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_dir = os.path.isdir(model_name)
    custom_ckpt = is_dir and os.path.isfile(os.path.join(model_name, "model.pt"))

    if custom_ckpt:
        # try to read num_labels (defaults to 1)
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
        return tokenizer, model, device, True

    # HF model path/name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    return tokenizer, model, device, False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _apply_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    return x / float(temperature)


def _rescale(values: np.ndarray, method: str) -> np.ndarray:
    method = (method or "none").lower()
    if method == "minmax":
        vmin, vmax = (
            float(values.min(initial=np.inf)),
            float(values.max(initial=-np.inf)),
        )
        rng = vmax - vmin
        if rng <= 1e-8:
            return np.full_like(values, 0.5, dtype=np.float32)
        return ((values - vmin) / rng).astype(np.float32)
    elif method == "zscore":
        mu, sigma = float(values.mean()), float(values.std())
        if sigma <= 1e-8:
            return np.full_like(values, 0.0, dtype=np.float32)
        return ((values - mu) / sigma).astype(np.float32)
    else:
        return values.astype(np.float32)


def _aggregate(values: np.ndarray, mode: str, topk: int = 3) -> float:
    mode = (mode or "mean").lower()
    if values.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "p25":
        return float(np.percentile(values, 25))
    if mode == "p75":
        return float(np.percentile(values, 75))
    if mode == "min":
        return float(np.min(values))
    if mode == "max":
        return float(np.max(values))
    if mode == "topk_mean":
        k = max(1, min(topk, values.size))
        idx = np.argpartition(values, -k)[-k:]
        return float(np.mean(values[idx]))
    return float(np.mean(values))


def compute_quality_scores(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    pair_texts: List[str] | None = None,
    use_auto: bool = True,
    debug_iqs: bool = False,
) -> np.ndarray:
    """Berechnet (logits oder probs) für (topic, argument)-Paare und gibt Wahrscheinlichkeiten in [0,1] zurück."""

    def dbg(*a, **k):
        if debug_iqs:
            print("[IQS-DEBUG]", *a, **k)

    if model_name is None:
        return np.full(len(texts), 0.5, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, device, is_custom = load_model(model_name, device=device)

    logits_all: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="IQS", unit="batch"):
        batch = texts[i : i + batch_size]
        batch_pair = pair_texts[i : i + batch_size] if pair_texts is not None else None
        if debug_iqs:
            try:
                dbg(f"Batch {i // batch_size + 1}: Größe={len(batch)}")
                dbg(
                    f"Beispiel-Paar 0: arg='{batch[0][:80]}{'...' if len(batch[0]) > 80 else ''}' | topic='{(batch_pair[0] if batch_pair else None)}'"
                )
            except Exception as e:
                dbg(f"[DEBUG] Tokenizer-Infos fehlgeschlagen: {e}")
        with torch.no_grad():
            enc = tokenizer(
                batch,  # argument text first
                batch_pair,  # topic as text_pair
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(model.device if hasattr(model, "device") else device)

            if is_custom:
                probs = (
                    model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                outputs = model(**enc)
                logits = outputs.logits
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                logits = logits.detach().cpu().numpy().astype(np.float32)
                probs = _sigmoid(logits)
        logits_all.extend(probs.tolist())

    probs = np.asarray(logits_all, dtype=np.float32)
    if debug_iqs and probs.size:
        dbg(
            f"Gesamt Probs: n={probs.size}, min={float(np.min(probs)):.4f}, max={float(np.max(probs)):.4f}, mean={float(np.mean(probs)):.4f}, std={float(np.std(probs)):.4f}"
        )
    return probs


# ---------------------------------------------------------------------------
# Topic-Vorverarbeitung
# ---------------------------------------------------------------------------


def _clean_topic_value(val):
    """Normalisiert Topic-Werte: '_' -> ' ', trimmt, entfernt Endpunkte, reduziert Multiple-Spaces."""
    if pd.isna(val):
        return val
    s = str(val)
    s = s.replace("_", " ")
    s = s.strip()
    s = s.rstrip(".")
    s = re.sub(r"\s+", " ", s)
    return s


def preprocess_topics(df: pd.DataFrame) -> pd.DataFrame:
    if "topic" not in df.columns:
        return df
    df = df.copy()
    df["topic"] = df["topic"].map(_clean_topic_value)
    return df


def group_by_topic(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "topic" in df.columns:
        return {
            t: sub.reset_index(drop=True) for t, sub in df.groupby("topic", sort=False)
        }
    return {"GLOBAL": df.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# Utility: Keypoint-Spalte erkennen
# ---------------------------------------------------------------------------
def _detect_keypoint_column(df: pd.DataFrame, preferred: str = "keypoint") -> str:
    """Finde die Spalte mit dem Keypoint-Text. Prüft mehrere gängige Namen."""
    if preferred in df.columns:
        return preferred
    candidates = [
        "keypoint",
        "key_point",
        "kp",
        "kp_text",
        "text",
        "content",
        "summary",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Konnte keine Keypoint-Textspalte finden. Erwartet z.B. eine der Spalten {candidates} oder nutze --keypoint-col."
    )


def compute_keypoint_iqs_for_df(
    df_kp: pd.DataFrame,
    keypoint_col: str,
    topic_col: str,
    qual_model_name: str,
    debug_iqs: bool = False,
) -> pd.DataFrame:
    """Berechnet IQS für *Keypoints* (pro Zeile) anhand (topic, keypoint_text)."""
    df = df_kp.copy()
    # Topic-Spalte sicherstellen
    if topic_col not in df.columns:
        df[topic_col] = "GLOBAL"
    else:
        df = preprocess_topics(df)
    if keypoint_col not in df.columns:
        # Auto-Detect versuchen
        keypoint_col = _detect_keypoint_column(df, preferred=keypoint_col)
    texts = df[keypoint_col].astype(str).fillna("").tolist()
    topics = df[topic_col].astype(str).fillna("").tolist()
    probs = compute_quality_scores(
        texts,
        qual_model_name,
        pair_texts=topics,
        debug_iqs=debug_iqs,
    )
    out = df.copy()
    # Normalisierte Keypoint-Spalte für Export/Details
    out["keypoint_text"] = df[keypoint_col].astype(str)
    out["IQS"] = probs
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Berechnet den durchschnittlichen IQS (Argumentqualitätswert) **nur** aus Keypoints. "
            "Es werden keine Argumente benötigt/berücksichtigt und es erfolgt keine Visualisierung. "
            "Die Ausgabe ist eine CSV mit den pro Topic gemittelten IQS-Werten je Keypoints-Datei."
        )
    )
    p.add_argument(
        "--keypoints",
        required=True,
        nargs="+",
        type=Path,
        help="Eine oder mehrere Keypoint-CSVs mit optionaler Spalte 'topic'.",
    )
    p.add_argument(
        "--quality-model",
        required=False,
        default="../../alshomary2021/models/argument-quality-ibm-reproduced/bert_wa",
        help="Pfad/Name des Quality-Modells (HuggingFace Ordner). Erwartet (topic, keypoint)-Input.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ziel-CSV-Datei mit den aggregierten IQS-Werten.",
    )
    p.add_argument(
        "--keypoint-col",
        default="keypoint",
        help="Name der Spalte mit dem Keypoint-Text in den Keypoint-CSVs.",
    )
    p.add_argument(
        "--topic-col",
        default="topic",
        help="Name der Topic-Spalte in den Keypoint-CSVs (falls nicht vorhanden, wird 'GLOBAL' genutzt).",
    )
    p.add_argument(
        "--debug-iqs", action="store_true", help="Zusätzliche Debug-Ausgaben."
    )
    return p.parse_args()


def main() -> None:
    print("Berechne durchschnittliche IQS-Werte ausschließlich aus Keypoints …")
    args = parse_args()

    all_topic_rows = []  # pro Datei je Topic aggregierte IQS

    for kp_path in args.keypoints:
        df_kp = pd.read_csv(kp_path)
        # Topic-Spalte ggf. umbenennen (für interne Konsistenz)
        if args.topic_col in df_kp.columns and args.topic_col != "topic":
            df_kp = df_kp.rename(columns={args.topic_col: "topic"})

        # IQS für Keypoints berechnen
        df_scored = compute_keypoint_iqs_for_df(
            df_kp,
            keypoint_col=args.keypoint_col,
            topic_col="topic" if "topic" in df_kp.columns else args.topic_col,
            qual_model_name=args.quality_model,
            debug_iqs=args.debug_iqs,
        )

        # Aggregation je Topic (immer Mittelwert)
        df_topics = aggregate_keypoint_iqs_by_topic(
            df_scored,
            topic_col="topic" if "topic" in df_scored.columns else args.topic_col,
            agg="mean",
            topk=3,
        )
        df_topics.insert(0, "file", str(kp_path))
        all_topic_rows.append(df_topics)

    # Zusammenführen und CSV schreiben
    if all_topic_rows:
        df_out = pd.concat(all_topic_rows, ignore_index=True)
    else:
        df_out = pd.DataFrame(columns=["file", "topic", "IQS", "n_keypoints"])

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Speichere nur die aggregierten Topic-IQS
    cols = [c for c in ["file", "topic", "IQS", "n_keypoints"] if c in df_out.columns]

    # Select only available columns using a list (not a tuple)
    df_out = df_out[cols]
    df_out.to_csv(out_path, index=False)
    print(f"Ergebnis geschrieben nach: {out_path}")


# ---------------------------------------------------------------------------
# Aggregation: Keypoint-IQS zu Topic-IQS (pro Topic)
# ---------------------------------------------------------------------------
def aggregate_keypoint_iqs_by_topic(
    df_scored: pd.DataFrame,
    topic_col: str = "topic",
    agg: str = "mean",
    topk: int = 3,
) -> pd.DataFrame:
    """Aggregiert Keypoint-IQS pro Topic zu einem Topic-IQS (z.B. Mittelwert)."""
    if topic_col not in df_scored.columns:
        df_scored = df_scored.copy()
        df_scored[topic_col] = "GLOBAL"
    groups = df_scored.groupby(topic_col, sort=False)
    rows = []
    for t, sub in groups:
        vals = sub["IQS"].dropna().astype(float).values
        iqs_topic = _aggregate(vals, agg, topk=topk) if vals.size else float("nan")
        rows.append({"topic": t, "IQS": iqs_topic, "n_keypoints": int(vals.size)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
