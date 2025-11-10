#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

try:
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForSequenceClassification,
        BertConfig,
        BertForSequenceClassification,
    )
    import torch
except ImportError as e:
    raise SystemExit(
        "Bitte installiere 'sentence-transformers' und 'transformers': "
        "pip install sentence-transformers transformers"
    ) from e

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embed_texts(
    texts: List[str], model: SentenceTransformer, batch_size: int = 64
) -> np.ndarray:
    """Embed a list of texts returning a (n, d) NumPy array."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(
            batch, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings.append(batch_emb)
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Qualitäts-Scores (für IQS01)
# ---------------------------------------------------------------------------


def compute_quality_scores(
    texts: List[str],
    model_name: str | None,
    batch_size: int = 32,
    use_auto: bool = True,
    pair_texts: List[str] | None = None,
    debug_iqs: bool = False,
) -> np.ndarray:
    """Return raw quality logits per text.
    If *model_name* is None, returns zeros.
    *use_auto* tries to load a full HF model via AutoModelForSequenceClassification; if that fails, falls back to BERT config+state_dict.
    """

    def dbg(*a, **k):
        if debug_iqs:
            print("[IQS-DEBUG]", *a, **k)

    if model_name is None:
        return np.zeros(len(texts), dtype=np.float32)

    dbg("Starte compute_quality_scores …")
    dbg(f"model_name={model_name} | batch_size={batch_size} | use_auto={use_auto}")
    dbg(f"Anzahl Texte={len(texts)} | pair_texts gesetzt={pair_texts is not None}")
    if pair_texts is not None:
        preview_pairs = list(zip(texts[:3], pair_texts[:3]))
        for i, (t, p) in enumerate(preview_pairs):
            dbg(
                f"Beispiel Paar {i}: (topic='{p}', argument='{t[:120]}{'…' if len(t) > 120 else ''}')"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg(f"Gerät: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Try robust HF loading first
    model = None
    tokenizer = None
    if use_auto:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            )
            model.to(device).eval()
            dbg("Ladeweg: AutoModelForSequenceClassification (lokale Dateien)")
            dbg(f"Geladenes Modell: {model.__class__.__name__}")
        except Exception:
            dbg(
                "AutoModel-Ladeweg fehlgeschlagen – wechsle zu Fallback (BERT + state_dict)"
            )
            model = None
            tokenizer = None

    if model is None:
        # Fallback: local folder with tokenizer + model.pt (custom state_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        except Exception:
            config = BertConfig.from_pretrained("bert-base-uncased")
        # Ensure a regression head with a single output to match the checkpoint
        config.num_labels = 1
        config.problem_type = "regression"
        model = BertForSequenceClassification(config)
        state = torch.load(f"{model_name}/model.pt", map_location="cpu")
        # Load strictly once shapes match; otherwise allow non-critical misses
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        dbg("Ladeweg: Fallback BERTForSequenceClassification mit custom state_dict")
        dbg(
            f"Config.num_labels={config.num_labels} | problem_type={getattr(config, 'problem_type', None)}"
        )

    def _maybe_to_prob(values: np.ndarray) -> np.ndarray:
        # If values already look like probabilities, keep them; otherwise apply sigmoid
        if values.size > 0 and np.all((values >= 0.0) & (values <= 1.0)):
            return values.astype(np.float32)
        return _sigmoid(values.astype(np.float32))

    dbg("Beginne Batch-Inferenz für IQS …")
    logits_all: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Quality", unit="batch"):
        start_idx, end_idx = i, min(i + batch_size, len(texts))
        batch = texts[i : i + batch_size]
        if pair_texts is not None:
            batch_pair = pair_texts[i : i + batch_size]
        else:
            batch_pair = None
        dbg(f"Batch {start_idx}:{end_idx} (Größe={len(batch)})")
        if batch_pair is not None and len(batch) > 0:
            dbg(
                f"Erstes (topic, argument): ('{batch_pair[0]}', '{batch[0][:200]}{'…' if len(batch[0]) > 200 else ''}')"
            )
        tokenized = tokenizer(
            batch_pair,  # Topic zuerst
            text_pair=batch,  # Argument als zweites Segment
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**tokenized)
            # Handle models that return logits in different shapes
            logits = outputs.logits
            dbg(f"Logits-Shape={tuple(logits.shape)}")
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            logits = logits.detach().cpu().numpy()
            if logits.ndim == 0:
                dbg(f"Logit: {float(logits)}")
            else:
                dbg(f"Logits (erste 5): {logits[:5].ravel()}")
        logits_all.extend(logits.tolist())
    dbg(f"Gesamtanzahl Logits: {len(logits_all)}")
    logits_arr = np.asarray(logits_all, dtype=np.float32)
    # Convert to probabilities if needed (avoid double-sigmoid if the model already outputs probs)
    dbg("Wandle ggf. in Wahrscheinlichkeiten um …")
    probs_arr = _maybe_to_prob(logits_arr)
    dbg(
        f"IQS-Probs (min/mean/max): {float(np.min(probs_arr)):.4f} / {float(np.mean(probs_arr)):.4f} / {float(np.max(probs_arr)):.4f}"
    )
    dbg(f"IQS-Probs (erste 10): {probs_arr[:10]}")
    dbg("Beende compute_quality_scores")
    return probs_arr


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
    # default
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# SUSWIR-Bestandteile
# ---------------------------------------------------------------------------


def redundancy_diversity(emb_k: np.ndarray) -> float:
    """RedundancyFree in [0,1]; höher = weniger redundant."""
    if len(emb_k) < 2:
        return 1.0
    sim = cosine_similarity(emb_k)
    idx = np.triu_indices_from(sim, k=1)
    if idx[0].size == 0:
        return 1.0
    mean_sim = float(sim[idx].mean())  # in [-1, 1]
    return float(1.0 - ((np.clip(mean_sim, -1.0, 1.0) + 1.0) / 2.0))


def argument_coverage_by_max(max_vals: np.ndarray, tau: float) -> float:
    """AC@τ: Anteil der Argumente, deren Maximalwert ≥ τ ist."""
    return float((max_vals >= tau).mean())


# ---------------------------------------------------------------------------
# Kernroutine: SBERT Variante für MMS & AC (vereinfacht)
# ---------------------------------------------------------------------------


def evaluate_topic_dual(
    args_df: pd.DataFrame,
    kp_df: pd.DataFrame,
    sbert: SentenceTransformer,
    qual_model_name: str | None,
    taus: List[float],
    suswir_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    iqs_temperature: float = 1.0,
    iqs_rescale: str = "none",
    iqs_agg: str = "mean",
    iqs_topk: int = 3,
    debug_iqs: bool = False,
) -> Dict[str, float]:
    """
    Berechnet:
      - MMS_sbert (Cosine; max Cosine pro Argument)
      - AC@τ_sbert (per-Argument Max)
      - RedundancyFree
      - IQS01
      - SUSWIR@τ (gewichtet, Standard 0.25 je RF, AC@τ, MMS, IQS)
    """
    arguments = args_df["argument"].tolist()
    keypoints = kp_df["key_point"].tolist()
    # Extract the single topic string for this group (after grouping by topic in main)
    if (
        "topic" in kp_df.columns
        and not kp_df["topic"].isna().all()
        and len(args_df) > 0
    ):
        # all rows in this group should share the same topic
        topic_values = kp_df["topic"].dropna().unique().tolist()
        topic_str = str(topic_values[0]) if len(topic_values) > 0 else ""
    else:
        topic_str = ""

    if debug_iqs:
        print("[IQS-DEBUG] —— Kontext für IQS ——")
        print(f"Topic: '{topic_str}'")
        print(f"#Arguments={len(arguments)} | #Keypoints={len(keypoints)}")
        pair_preview = list(zip([topic_str] * min(3, len(arguments)), arguments[:3]))
        for j, (tp, ar) in enumerate(pair_preview):
            print(
                f"[IQS-DEBUG] Beispiel Input-Paar {j}: (topic='{tp}', argument='{ar[:200]}{'…' if len(ar) > 200 else ''}')"
            )
        print("[IQS-DEBUG] —— Ende Kontext ——")

    # Embeddings für Cosine-basierte Teile
    emb_args = embed_texts(arguments, sbert)
    emb_kps = embed_texts(keypoints, sbert)

    sim_matrix = cosine_similarity(emb_args, emb_kps)  # (|A|, |K|)
    max_sim_sbert = sim_matrix.max(axis=1)  # (|A|,)

    # Komponenten
    r_free = redundancy_diversity(emb_kps)
    mms_sbert = float(max_sim_sbert.mean())

    if qual_model_name is not None:
        # Build (topic, argument) pairs: same topic string for all arguments in this group
        topics_for_args = [topic_str] * len(arguments)
        # Get probabilities in [0,1] from the quality model
        q_probs = compute_quality_scores(
            arguments, qual_model_name, pair_texts=topics_for_args, debug_iqs=debug_iqs
        )
        # Apply temperature on logit scale and map back to prob space
        if iqs_temperature != 1.0:
            eps = 1e-6
            p = np.clip(q_probs, eps, 1 - eps)
            logit = np.log(p / (1 - p))
            logit = _apply_temperature(logit, iqs_temperature)
            q_probs = _sigmoid(logit)
            if debug_iqs:
                print(f"[IQS-DEBUG] Temperature={iqs_temperature} angewandt")
                print(
                    f"[IQS-DEBUG] Nach Temperature (min/mean/max): {float(np.min(q_probs)):.4f} / {float(np.mean(q_probs)):.4f} / {float(np.max(q_probs)):.4f}"
                )
        # Optional per-topic rescaling and aggregation over arguments of this topic
        q_probs = _rescale(q_probs, iqs_rescale)
        if debug_iqs:
            print(f"[IQS-DEBUG] Rescale-Methode: {iqs_rescale}")
            print(
                f"[IQS-DEBUG] Nach Rescale (min/mean/max): {float(np.min(q_probs)):.4f} / {float(np.mean(q_probs)):.4f} / {float(np.max(q_probs)):.4f}"
            )
        iqs01 = _aggregate(q_probs, iqs_agg, topk=iqs_topk)
        if debug_iqs:
            print(f"[IQS-DEBUG] Aggregation: mode={iqs_agg}, topk={iqs_topk}")
            print(
                f"[IQS-DEBUG] Finaler IQS01 für Topic '{topic_str}': {float(iqs01):.6f}"
            )
    else:
        iqs01 = 0.5  # neutral

    results: Dict[str, float] = {
        "RF": r_free,
        "MMS": mms_sbert,
        "IQS": iqs01,
    }

    for tau in taus:
        # AC@τ (per-Argument Max)
        ac_sbert = argument_coverage_by_max(max_sim_sbert, tau)
        results[f"AC@{tau}"] = ac_sbert
        wR, wAC, wMMS, wIQS = suswir_weights
        suswir = (wR * r_free) + (wAC * ac_sbert) + (wMMS * mms_sbert) + (wIQS * iqs01)
        results[f"SUSWIR@{tau}"] = float(suswir)

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clean_topic_value(val):
    """Normalisiert Topic-Werte: ersetzt Unterstriche durch Leerzeichen,
    trimmt Whitespace, entfernt Punkt(e) am Ende und reduziert Mehrfach-Leerzeichen."""
    if pd.isna(val):
        return val
    s = str(val)
    s = s.replace("_", " ")
    s = s.strip()
    # alle abschließenden Punkte entfernen
    s = s.rstrip(".")
    # auf einfache Leerzeichen normalisieren
    s = re.sub(r"\s+", " ", s)
    return s


def preprocess_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt eine Kopie von *df* zurück, in der die Spalte 'topic' bereinigt wurde."""
    if "topic" not in df.columns:
        return df
    df = df.copy()
    df["topic"] = df["topic"].map(_clean_topic_value)
    return df


def group_by_topic(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """topic ➜ Teil-DataFrame (oder GLOBAL)."""
    if "topic" in df.columns:
        return {
            t: sub.reset_index(drop=True) for t, sub in df.groupby("topic", sort=False)
        }
    return {"GLOBAL": df.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate redundancy (RF), MMS (SBERT), AC@τ (SBERT), IQS01 (from a topic-aware quality model using (topic, argument) pairs), and SUSWIR@τ (weighted combo) — without entailment or AC_density. "
            "Controls: --iqs-temperature, --iqs-rescale, --iqs-agg, --iqs-topk to adjust IQS dynamic range."
        )
    )
    p.add_argument(
        "--arguments", required=True, type=Path, help="CSV mit Spalte 'argument'."
    )
    p.add_argument(
        "--keypoints", required=True, type=Path, help="CSV mit Spalte 'key_point'."
    )
    p.add_argument(
        "--sbert-model",
        default="../../Alshomary2021/models/roberta-large-final-model-fold-4-2023-07-05_16-02-50",
        help="Sentence-BERT Modellname oder Pfad.",
    )
    p.add_argument(
        "--quality-model",
        default="../../Alshomary2021/models/argument-quality-ibm-reproduced/bert_wa",
        help="Optionaler Quality-Schätzer (für IQS01). Erwartet Topic+Argument als Input (z. B. Webis-Reproduktion).",
    )
    p.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.7],
        help="Schwellen τ für AC@τ.",
    )
    p.add_argument(
        "--suswir-weights",
        nargs=4,
        type=float,
        metavar=("W_RED", "W_AC", "W_MMS", "W_IQS"),
        default=[0.25, 0.25, 0.25, 0.25],
        help=(
            "Gewichte für SUSWIR: W_RED*RF + W_AC*AC@τ + W_MMS*MMS + W_IQS*IQS. "
            "Standard: 0.25 0.25 0.25 0.25"
        ),
    )
    p.add_argument(
        "--iqs-temperature",
        type=float,
        default=1.0,
        help="Temperature <1.0 sharpens (more spread), >1.0 flattens the IQS sigmoid.",
    )
    p.add_argument(
        "--iqs-rescale",
        choices=["none", "minmax", "zscore"],
        default="none",
        help="Optional per-topic rescaling of IQS scores to increase dynamic range.",
    )
    p.add_argument(
        "--iqs-agg",
        choices=["mean", "median", "p25", "p75", "min", "max", "topk_mean"],
        default="mean",
        help="Aggregation over argument IQS scores per topic.",
    )
    p.add_argument(
        "--iqs-topk",
        type=int,
        default=3,
        help="k for --iqs-agg topk_mean.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Pfad für die CSV-Ausgabe. Wenn nicht gesetzt: ./output/<ARGS>__<KPS>_results.csv",
    )
    p.add_argument(
        "--debug-iqs",
        action="store_true",
        help="Ausführliche Debug-Prints nur für die IQS-Berechnung ausgeben",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_args = pd.read_csv(args.arguments)
    df_kps = pd.read_csv(args.keypoints)

    # Topics vor dem Grouping bereinigen
    df_args = preprocess_topics(df_args)
    df_kps = preprocess_topics(df_kps)

    if "argument" not in df_args.columns or "key_point" not in df_kps.columns:
        raise ValueError(
            "Beide CSVs benötigen Spalten 'argument' (Args) und 'key_point' (KPs)."
        )

    arg_groups = group_by_topic(df_args)
    kp_groups = group_by_topic(df_kps)

    missing_topics = set(arg_groups) - set(kp_groups)
    if missing_topics:
        raise ValueError(
            f"Keine Keypoints für Topics: {', '.join(sorted(missing_topics))}."
        )

    # Modelle laden
    sbert = SentenceTransformer(args.sbert_model)

    per_topic_results: Dict[str, Dict[str, float]] = {}
    for topic, arg_df in arg_groups.items():
        kp_df = kp_groups[topic]
        per_topic_results[topic] = evaluate_topic_dual(
            arg_df,
            kp_df,
            sbert,
            args.quality_model,
            taus=args.thresholds,
            suswir_weights=tuple(args.suswir_weights),
            iqs_temperature=args.iqs_temperature,
            iqs_rescale=args.iqs_rescale,
            iqs_agg=args.iqs_agg,
            iqs_topk=args.iqs_topk,
            debug_iqs=args.debug_iqs,
        )

    # Ausgabepfad bestimmen: falls kein --output angegeben, automatisch aus Dateinamen bauen
    if args.output is None:
        a_stem = Path(args.arguments).stem
        k_stem = Path(args.keypoints).stem
        output_dir = Path("./output")
        output_path = output_dir / f"{a_stem}VS{k_stem}.csv"
    else:
        output_path = args.output
        output_dir = output_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for topic, metrics in per_topic_results.items():
        row = {"topic": topic}
        row.update(metrics)
        rows.append(row)
    df_output = pd.DataFrame(rows)
    df_output.to_csv(output_path, index=False)
    print(f"Ergebnis geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()
