#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Bitte installiere 'sentence-transformers': pip install sentence-transformers"
    ) from e


# ---------------------------------------------------------------------------
# Embeddings (identisch zum Originalverhalten)
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
# AC@τ
# ---------------------------------------------------------------------------


def argument_coverage_by_max(max_vals: np.ndarray, tau: float) -> float:
    """AC@τ: Anteil der Argumente, deren Maximalwert ≥ τ ist."""
    return float((max_vals >= tau).mean())


# ---------------------------------------------------------------------------
# Topic-Preprocessing & Grouping (wie im Original)
# ---------------------------------------------------------------------------


def _clean_topic_value(val):
    """Normalisiert Topic-Werte: ersetzt Unterstriche durch Leerzeichen,
    trimmt Whitespace, entfernt Punkt(e) am Ende und reduziert Mehrfach-Leerzeichen."""
    if pd.isna(val):
        return val
    s = str(val)
    s = s.replace("_", " ")
    s = s.strip()
    s = s.rstrip(".")  # alle abschließenden Punkte entfernen
    s = re.sub(r"\s+", " ", s)  # auf einfache Leerzeichen normalisieren
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
# AC-Evaluierung (nur SBERT + AC, keine MMS/Redundancy/IQS)
# ---------------------------------------------------------------------------


def evaluate_ac_only_per_topic(
    args_df: pd.DataFrame,
    kps_df: pd.DataFrame,
    sbert: SentenceTransformer,
    taus: List[float],
) -> Dict[str, float]:
    """Berechnet nur AC@τ (über SBERT-Cosine-Max pro Argument) je Topic."""
    arguments = args_df["argument"].tolist()
    keypoints = kps_df["key_point"].tolist()

    # SBERT-Embeddings
    emb_args = embed_texts(arguments, sbert)
    emb_kps = embed_texts(keypoints, sbert)

    # Cosine-Similarity & Max pro Argument
    sim_matrix = cosine_similarity(emb_args, emb_kps)  # (|A|, |K|)
    max_sim_per_arg = sim_matrix.max(axis=1)  # (|A|,)

    results: Dict[str, float] = {}
    for tau in taus:
        results[f"AC@{tau}"] = argument_coverage_by_max(max_sim_per_arg, tau)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Berechnet ausschließlich AC@τ (Argument Coverage) pro Topic auf Basis von SBERT-Cosine-Maxima. "
            "Unterstützt eine oder mehrere Keypoint-CSVs und schreibt alle Ergebnisse in eine gemeinsame Ausgabe-CSV."
        )
    )
    p.add_argument(
        "--arguments", required=True, type=Path, help="CSV mit Spalte 'argument'."
    )
    p.add_argument(
        "--keypoints",
        required=True,
        nargs="+",
        type=Path,
        help="Eine oder mehrere CSVs mit Spalte 'key_point'.",
    )
    p.add_argument(
        "--sbert-model",
        default="../../khosravani2024/models/V1",
        help="Sentence-BERT Modellname oder Pfad (beibehalten wie im Original).",
    )
    p.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.7],
        help="Schwellen τ für AC@τ.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default="./output/ac_results.csv",
        help="Pfad für die CSV-Ausgabe. Wenn nicht gesetzt: ./output/<ARGS>VS<KPS>_ac.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_args = pd.read_csv(args.arguments)

    # Topics wie im Original bereinigen
    df_args = preprocess_topics(df_args)

    if "argument" not in df_args.columns:
        raise ValueError("Die Arguments-CSV benötigt die Spalte 'argument'.")

    arg_groups = group_by_topic(df_args)

    # SBERT laden (Model/Path beibehalten)
    sbert = SentenceTransformer(args.sbert_model)

    # Ergebnisse über mehrere Keypoint-Dateien sammeln
    rows = []

    for kp_path in args.keypoints:
        df_kps = pd.read_csv(kp_path)
        df_kps = preprocess_topics(df_kps)

        if "key_point" not in df_kps.columns:
            raise ValueError(
                f"Die Keypoints-CSV '{kp_path}' benötigt die Spalte 'key_point'."
            )

        kp_groups = group_by_topic(df_kps)

        missing_topics = set(arg_groups) - set(kp_groups)
        if missing_topics:
            raise ValueError(
                f"Keine Keypoints für Topics in '{kp_path}': {', '.join(sorted(missing_topics))}."
            )

        per_topic_results: Dict[str, Dict[str, float]] = {}
        for topic, arg_df in arg_groups.items():
            kps_df = kp_groups[topic]
            per_topic_results[topic] = evaluate_ac_only_per_topic(
                arg_df, kps_df, sbert, taus=args.thresholds
            )

        # In Zeilenform bringen und Quelle der KPs mitschreiben
        for topic, metrics in per_topic_results.items():
            row = {"topic": topic, "keypoints_file": str(kp_path)}
            row.update(metrics)
            rows.append(row)

    # Ausgabepfad
    if args.output is None:
        a_stem = Path(args.arguments).stem
        k_stem = Path(args.keypoints).stem
        output_dir = Path("./output")
        output_path = output_dir / f"{a_stem}VS{k_stem}_ac.csv"
    else:
        output_path = args.output
        output_dir = output_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Schreiben
    df_out = pd.DataFrame(rows)
    df_out = df_out[["keypoints_file", "topic", "AC@0.7"]].rename(
        columns={"AC@0.7": "AC", "keypoints_file": "file"}
    )
    df_out.to_csv(output_path, index=False)
    print(f"AC-Ergebnis geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()
