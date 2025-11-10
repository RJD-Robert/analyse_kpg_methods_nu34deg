#!/usr/bin/env python
"""
paper_evaluation.py
==================
Berechnet zentrale Evaluationsmetriken (ROUGE-1/2/L-F, BLEURT-basierte Soft-Precision/Recall/F1
sowie die Coverage-Metrik nach Khosravani et al.) für Key-Point-Listen.

Eingaben
--------
* Zwei CSV-Dateien – je eine Zeile pro Key Point mit den Spalten
  - ``topic``  (Topic-ID oder -Titel)
  - ``key_point``  (Text des Key Points)

  ``system.csv``  : von Deinem Modell generierte Key Points
  ``reference.csv`` : goldene/Referenz-Key Points

Ausgabe
-------
Metriken über alle Topics gemittelt (ungewichtet), ausgegeben auf STDOUT.

Nutzung (Beispiel)
------------------
```
python paper_evaluation.py --system_csv sys.csv --reference_csv ref.csv
```

Benötigte Libraries
-------------------
```
pip install pandas rouge-score evaluate transformers torch
```

Hinweis: BLEURT- und RoBERTa-Modelle werden beim ersten Aufruf automatisch
aus dem HuggingFace-Hub heruntergeladen.
"""

import argparse
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from rouge_score import rouge_scorer
import evaluate
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sys
import time
import os
import numpy as np

# Vordefinierte drei Test-Topics (KPA-2021 Shared Task)
TEST_TOPICS = [
    "Routine child vaccinations should be mandatory",
    "Social media platforms should be regulated by the government",
    "The USA is a good country to live in",
]

###############################################################################
# Daten laden
###############################################################################


def load_kps(path: str) -> Dict[str, List[str]]:
    print(f"[{time.strftime('%H:%M:%S')}] [load_kps] Reading CSV from {path}")
    df = pd.read_csv(path)
    if not {"topic", "key_point"}.issubset(df.columns):
        raise ValueError("CSV muss die Spalten 'topic' und 'key_point' enthalten.")
    df = df.dropna(subset=["key_point"]).astype({"topic": str, "key_point": str})
    grouped: Dict[str, List[str]] = defaultdict(list)
    print(f"[{time.strftime('%H:%M:%S')}] [load_kps] Grouping key points by topic…")
    for _, row in df.iterrows():
        if _ % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] [load_kps] Processed {_} rows")
        # Normalize topic by replacing underscores with spaces and removing trailing dots
        topic = row["topic"].strip().rstrip(".").replace("_", " ")
        grouped[topic].append(row["key_point"].strip())
    print(
        f"[{time.strftime('%H:%M:%S')}] [load_kps] Finished. Loaded {len(grouped)} topics."
    )
    for topic_name, kps in grouped.items():
        print(f"[DEBUG load_kps] Topic '{topic_name}': {len(kps)} key points")
    return grouped


###############################################################################
# ROUGE 1/2/L F
###############################################################################


def compute_rouge(
    sys_topics: Dict[str, List[str]], ref_topics: Dict[str, List[str]]
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_rouge] Starting ROUGE computation for {len(ref_topics)} topics."
    )
    r1, r2, rl = [], [], []
    for topic in ref_topics:
        print(
            f"[{time.strftime('%H:%M:%S')}] [compute_rouge] Processing topic '{topic}'"
        )
        if topic not in sys_topics:
            continue  # Topic in Gold, aber nicht im System – skip
        ref_str = "\n".join(ref_topics[topic])
        sys_str = "\n".join(sys_topics[topic])
        scores = scorer.score(ref_str, sys_str)
        print(
            f"[DEBUG compute_rouge] Topic '{topic}': rouge1={scores['rouge1'].fmeasure:.4f}, rouge2={scores['rouge2'].fmeasure:.4f}, rougeL={scores['rougeL'].fmeasure:.4f}"
        )
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_rouge] Completed. Avg ROUGE‑1: {sum(r1) / len(r1) if r1 else 0:.4f}"
    )
    # Guard against no common topics to avoid division by zero
    if not r1:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {
        "rouge1": sum(r1) / len(r1),
        "rouge2": sum(r2) / len(r2),
        "rougeL": sum(rl) / len(rl),
    }


###############################################################################
# BLEURT-basierte Soft-Precision/Recall/F1
###############################################################################
_bleurt = None  # global Lazy-Loader
# Cache for pairwise BLEURT scores to avoid recomputation across topics
_bleurt_pair_cache: Dict[str, float] = {}


def _bleurt_scorer():
    global _bleurt
    if _bleurt is None:
        _bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")
    return _bleurt


def _soft_scores(sys_kps: List[str], ref_kps: List[str]):
    """
    Berechnet Soft-Precision/Recall/F1 wie bei Lee:
    - Reihen = System-KPs (Kandidaten / predictions)
    - Spalten = Referenz-KPs (references)
    - sP = Zeilenweises Max, dann Mittelwert (row-max mean)
    - sR = Spaltenweises Max, dann Mittelwert (col-max mean)
    """
    bleurt = _bleurt_scorer()
    print(
        f"[{time.strftime('%H:%M:%S')}] [_soft_scores] Calculating BLEURT matrix (sys x ref)"
    )

    # Paare: (System-KP, Referenz-KP) — genau wie bei Lee (Predictions = System, References = Gold)
    # WICHTIG: Nicht vertauschen; BLEURT ist nicht symmetrisch.
    pairs_sys_ref = [
        (k, a) for k in sys_kps for a in ref_kps
    ]  # predictions=sys, references=ref

    # Nur fehlende Paare berechnen; Cache-Key: "sys-kp - ref-kp"
    missing = [
        (k, a) for (k, a) in pairs_sys_ref if f"{k}-{a}" not in _bleurt_pair_cache
    ]
    print(
        f"[{time.strftime('%H:%M:%S')}] [_soft_scores] BLEURT pairs total: {len(pairs_sys_ref)}, missing: {len(missing)}"
    )
    if missing:
        preds = [k for (k, _) in missing]  # predictions = System-KPs
        refs = [a for (_, a) in missing]  # references  = Referenz-KPs
        scores = bleurt.compute(predictions=preds, references=refs)["scores"]
        for (k, a), s in zip(missing, scores):
            _bleurt_pair_cache[f"{k}-{a}"] = s

    # Score-Matrix: Zeilen = sys_kps, Spalten = ref_kps
    score_matrix = [
        [_bleurt_pair_cache.get(f"{k}-{a}", 0.0) for a in ref_kps] for k in sys_kps
    ]
    score_matrix = np.array(score_matrix).reshape(len(sys_kps), len(ref_kps))
    print(f"[DEBUG _soft_scores] score_matrix shape: {score_matrix.shape}")

    sp = (
        score_matrix.max(axis=-1).mean().tolist() if score_matrix.size else 0.0
    )  # row-max mean
    sr = (
        score_matrix.max(axis=0).mean().tolist() if score_matrix.size else 0.0
    )  # col-max mean
    sF1 = 0.0 if sp + sr == 0 else 2 * sp * sr / (sp + sr)

    print(
        f"[{time.strftime('%H:%M:%S')}] [_soft_scores] Soft-Precision (row-max mean): {sp:.4f}"
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] [_soft_scores] Soft-Recall (col-max mean): {sr:.4f}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] [_soft_scores] Soft-F1: {sF1:.4f}")
    return sp, sr, sF1


def compute_soft_metrics(
    sys_topics: Dict[str, List[str]], ref_topics: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Computes Li et al.'s soft‑Precision, soft‑Recall and soft‑F1 in the
    *micro‑averaged* style described in Equation (4) of the paper:

        • Concatenate *all* system KPs across topics into one list
        • Concatenate *all* reference KPs across topics into one list
        • Build a single BLEURT score matrix (sys × ref)
        • sP  = mean of row‑wise maxima
        • sR  = mean of column‑wise maxima
        • sF1 = 2·sP·sR / (sP + sR)

    This global computation ensures sP, sR and sF1 are internally
    consistent (their values obey the F‑harmonic relationship) and
    matches the evaluation protocol used in Li et al. (2024).
    """
    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_soft_metrics] Computing global soft metrics"
    )

    # Flatten KPs across all topics (micro‑average)
    sys_kps = [kp for kps in sys_topics.values() for kp in kps]
    ref_kps = [kp for kps in ref_topics.values() for kp in kps]

    sP, sR, sF1 = _soft_scores(sys_kps, ref_kps)

    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_soft_metrics] Completed. "
        f"softP={sP:.4f}, softR={sR:.4f}, softF1={sF1:.4f}"
    )

    return {"softP": sP, "softR": sR, "softF1": sF1}


###############################################################################
# Coverage-Metrik (Khosravani et al.)
###############################################################################


def _load_entailment_model(model_name: str):
    """
    Loads a binary entailment/MATCH classifier.

    It first tries to pull both model weights *and* tokenizer from
    `model_name`. If the tokenizer is missing (typical for fine‑tuned
    folders that contain only `pytorch_model.bin` and `config.json`),
    the function falls back to the standard `bert-base-uncased`
    tokenizer while still using the fine‑tuned weights.

    This mirrors the tolerant loading logic used in the KPG pipeline.
    """
    try:
        # Preferred path: model and tokenizer live together
        return pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,
        )
    except (OSError, TypeError, ValueError, ImportError) as e:
        # Fallback path: borrow a tokenizer from bert-base-uncased
        print(
            f"[warn] Tokenizer not found in '{model_name}'. "
            "Falling back to 'bert-base-uncased'. "
            f"Original error: {e}"
        )
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("text-classification", model=mdl, tokenizer=tok, device=-1)


def _coverage(sys_kps: List[str], ref_kps: List[str], clf):
    print(
        f"[{time.strftime('%H:%M:%S')}] [_coverage] Checking coverage for {len(ref_kps)} reference KPs"
    )
    matched = set()
    for idx, rkp in enumerate(ref_kps):
        if idx % 10 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] [_coverage] Processed {idx} reference KPs"
            )
        for gkp in sys_kps:
            # Klassische NLI-Pipeline akzeptiert Tuple → hier als joined-String
            result = clf(f"{rkp} [SEP] {gkp}", truncation=True)[0]
            print(
                f"[DEBUG _coverage] Comparing ref idx {idx} to sys KP: label={result['label']}, score={result.get('score', 'N/A')}"
            )
            # Normalisiere Labelnamen auf Kleinbuchstaben
            label = result["label"].lower()

            # Akzeptiere auch generische Hugging‑Face‑Bezeichnungen für die
            # positive Klasse.  „label_1“ (bzw. „1“) steht in Standard‑Exports
            # meist für „match/entail“.  Damit wird das Skript robuster, falls
            # das feingetunte Modell keine id2label‑Mapping enthält.
            if (
                label.startswith("entail")
                or label.startswith("match")
                or label in ("label_1", "1")
            ):
                matched.add(idx)
                break
    print(
        f"[{time.strftime('%H:%M:%S')}] [_coverage] Coverage: {len(matched)}/{len(ref_kps)} = {len(matched) / len(ref_kps) if ref_kps else 0:.4f}"
    )
    print(f"[DEBUG _coverage] Matched references: {len(matched)}/{len(ref_kps)}")
    return len(matched) / len(ref_kps) if ref_kps else 0.0


def compute_coverage(
    sys_topics: Dict[str, List[str]], ref_topics: Dict[str, List[str]], model_name: str
) -> Dict[str, float]:
    clf = _load_entailment_model(model_name)
    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_coverage] Loaded entailment model '{model_name}'"
    )
    covs = []
    for t in ref_topics:
        print(
            f"[{time.strftime('%H:%M:%S')}] [compute_coverage] Calculating coverage for topic '{t}'"
        )
        if t not in sys_topics:
            continue
        covs.append(_coverage(sys_topics[t], ref_topics[t], clf))
    print(
        f"[{time.strftime('%H:%M:%S')}] [compute_coverage] Avg coverage: {sum(covs) / len(covs) if covs else 0:.4f}"
    )
    # Guard against no common topics to avoid division by zero
    if not covs:
        return {"coverage": 0.0}
    return {"coverage": sum(covs) / len(covs)}


###############################################################################
# CLI
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Berechnet KP-Evaluationsmetriken aus zwei CSV-Dateien."
    )
    parser.add_argument("--system_csv", required=True, help="CSV mit System-Key-Points")
    parser.add_argument(
        "--reference_csv", required=True, help="CSV mit Referenz-Key-Points"
    )
    parser.add_argument(
        "--entail_model",
        default="../../khosravani2024/models/2",
        help="HF-Model für Coverage (fein-getuntes ArgKP-Modell empfohlen)",
    )
    parser.add_argument(
        "--kpa_test_only",
        action="store_true",
        help=(
            "Wenn gesetzt, werden nur die drei vordefinierten KPA-Test-Topics "
            "evaluiert: 'Routine child vaccinations should be mandatory', "
            "'Social media platforms should be regulated by the government', "
            "'The USA is a good country to live in'."
        ),
    )
    args = parser.parse_args()
    print(f"[{time.strftime('%H:%M:%S')}] [main] Starting evaluation script")

    sys_topics = load_kps(args.system_csv)
    print(
        f"[{time.strftime('%H:%M:%S')}] [main] System topics loaded: {len(sys_topics)}"
    )
    ref_topics = load_kps(args.reference_csv)
    print(
        f"[{time.strftime('%H:%M:%S')}] [main] Reference topics loaded: {len(ref_topics)}"
    )
    # Skip topics that appear only in one of the files
    common_topics = set(sys_topics).intersection(ref_topics)
    print(f"[{time.strftime('%H:%M:%S')}] [main] Common topics: {len(common_topics)}")
    # Optional: Nur die drei vordefinierten Test-Topics evaluieren
    if args.kpa_test_only:
        desired = {t.replace("_", " ") for t in TEST_TOPICS}
        before_n = len(common_topics)
        common_topics = common_topics.intersection(desired)
        print(
            f"[{time.strftime('%H:%M:%S')}] [main] --kpa_test_only aktiv. "
            f"Gewünschte Topics: {', '.join(TEST_TOPICS)}"
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] [main] Gefundene/übereinstimmende Topics: "
            f"{len(common_topics)}/{before_n}: {', '.join(sorted(common_topics)) if common_topics else '—'}"
        )
        if not common_topics:
            print(
                "Fehler: Keines der drei Test-Topics kommt in beiden CSV-Dateien vor.",
                file=sys.stderr,
            )
            sys.exit(1)
    if not common_topics:
        print(
            "Fehler: Keine gemeinsamen Topics zwischen System und Referenz gefunden.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys_topics = {topic: sys_topics[topic] for topic in common_topics}
    ref_topics = {topic: ref_topics[topic] for topic in common_topics}

    results = {}

    results.update(compute_coverage(sys_topics, ref_topics, args.entail_model))
    print(f"[{time.strftime('%H:%M:%S')}] [main] Coverage results added")
    results.update(compute_rouge(sys_topics, ref_topics))
    print(f"[{time.strftime('%H:%M:%S')}] [main] ROUGE results added")
    results.update(compute_soft_metrics(sys_topics, ref_topics))
    print(f"[{time.strftime('%H:%M:%S')}] [main] Soft metrics results added")

    print(f"[{time.strftime('%H:%M:%S')}] [main] Printing aggregated scores")
    print("\n=== Aggregierte Scores ===")
    for k, v in results.items():
        print(f"[{time.strftime('%H:%M:%S')}] [main] {k}: {v:.4f}")
        print(f"{k:10s}: {v:.4f}")
    # Save aggregated results to CSV
    ref_base = os.path.splitext(os.path.basename(args.reference_csv))[0]
    sys_base = os.path.splitext(os.path.basename(args.system_csv))[0]
    output_file = f"output/{ref_base}VS{sys_base}.csv"
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] [main] Results saved to {output_file}")


if __name__ == "__main__":
    main()
