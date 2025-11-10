#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as e:
    raise SystemExit(
        "Bitte installiere 'sentence-transformers': pip install sentence-transformers"
    ) from e


# --------- Bi-Encoder MMS ---------


def compute_mms_biencoder(
    arguments: List[str],
    keypoints: List[str],
    bi_encoder: SentenceTransformer,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
) -> float:
    """
    MMS (Bi-Encoder): Für jedes Argument den maximalen Cosinus-Score zu allen Keypoints.
    Wir encoden Argumente & Keypoints separat (einmal pro Datei), berechnen Cosinus-Ähnlichkeiten,
    nehmen pro Argument das Maximum und mitteln darüber.
    """
    if len(arguments) == 0 or len(keypoints) == 0:
        return float("nan")

    # Embeddings
    arg_emb = bi_encoder.encode(
        arguments,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    kp_emb = bi_encoder.encode(
        keypoints,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )

    # Cosinus-Ähnlichkeiten (|A| x |K|)
    sim = util.cos_sim(arg_emb, kp_emb)  # Tensor

    # pro Argument das Maximum über alle Keypoints
    max_per_arg = sim.max(dim=1).values

    # Mittelwert der Maxima
    return float(max_per_arg.mean().item())


# --------- Topic-Utilities (optional) ---------
def _clean_topic_value(val):
    if pd.isna(val):
        return val
    s = str(val).replace("_", " ").strip()
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


# --------- CLI ---------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Berechnet den MMS(Bi-Encoder)-Wert (Mean of Max Similarities) – optional pro Topic. "
        "Unterstützt mehrere Keypoint-CSV-Dateien, erzeugt kombinierte CSV und Visualisierung."
    )
    p.add_argument(
        "--arguments",
        required=True,
        type=Path,
        help="CSV mit Spalte 'argument' (+optional 'topic').",
    )
    p.add_argument(
        "--keypoints",
        required=True,
        type=Path,
        nargs="+",
        help="Eine oder mehrere CSV-Dateien mit Spalte 'key_point' (+optional 'topic').",
    )
    p.add_argument(
        "--model",
        required=False,
        default="../../khosravani2024/models/V1",
        help="Pfad oder Name des SentenceTransformer-Modells (Bi-Encoder).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default="./output/all_mms.csv",
        help="Pfad für die kombinierte CSV-Ausgabe über alle Keypoint-Dateien.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_args = pd.read_csv(args.arguments)
    print(f"Lese Arguments-CSV: {args.arguments}, {len(df_args)} Zeilen geladen.")

    # Topics (falls vorhanden) vorbereiten & gruppieren
    df_args = preprocess_topics(df_args)
    arg_groups = group_by_topic(df_args)
    print(f"Topics in Arguments-CSV: {list(arg_groups.keys())}")

    print(f"Lade Bi-Encoder Modell: {args.model}")
    model = SentenceTransformer(args.model)
    mms_fn = compute_mms_biencoder

    # Sammeln aller Ergebnisse über alle Keypoint-Dateien
    all_rows = []

    # Output-Basisverzeichnis bestimmen
    if args.output is None:
        a_stem = Path(args.arguments).stem
        output_dir = Path("./output")
        combined_output_path = output_dir / f"{a_stem}_ALL_mms.csv"
    else:
        combined_output_path = args.output
        output_dir = combined_output_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Für jede Keypoint-CSV berechnen wir pro Topic den MMS-Wert
    for kp_path in args.keypoints:
        print(f"Verarbeite Keypoints-Datei: {kp_path}")
        df_kps = pd.read_csv(kp_path)
        print(f" - {len(df_kps)} Zeilen in Keypoints-CSV geladen.")
        if "key_point" not in df_kps.columns:
            raise ValueError("Die Keypoints-CSV benötigt eine Spalte 'key_point'.")

        df_kps = preprocess_topics(df_kps)
        kp_groups = group_by_topic(df_kps)
        print(f" - Topics in Keypoints-CSV: {list(kp_groups.keys())}")

        # Themenmenge bestimmen (Schnittmenge)
        topics_args = set(arg_groups.keys())
        topics_kps = set(kp_groups.keys())
        common_topics = sorted(topics_args & topics_kps)
        missing_for_kps = sorted(topics_args - topics_kps)
        if missing_for_kps:
            print(
                f"Warnung: In '{kp_path}' fehlen Keypoints für Topics: {', '.join(missing_for_kps)}. Diese Topics werden übersprungen."
            )
        if not common_topics:
            raise ValueError(
                f"Keine gemeinsamen Topics zwischen Arguments und '{kp_path}'."
            )

        print(f"   Berechne MMS für {len(common_topics)} gemeinsame Topics...")
        # Pro Topic MMS berechnen
        rows = []
        for topic in common_topics:
            arg_df = arg_groups[topic]
            kp_df = kp_groups[topic]
            arguments = arg_df["argument"].astype(str).tolist()
            keypoints = kp_df["key_point"].astype(str).tolist()
            print(
                f"    -> Topic '{topic}': {len(arguments)} Argumente, {len(keypoints)} Keypoints"
            )
            mms = mms_fn(arguments, keypoints, model)
            rows.append({"file": Path(kp_path).name, "topic": topic, "MMS": mms})

        # Per-File-CSV
        a_stem = Path(args.arguments).stem
        k_stem = Path(kp_path).stem
        per_file_path = output_dir / f"{a_stem}VS{k_stem}_mms.csv"
        pd.DataFrame(rows).to_csv(per_file_path, index=False)
        print(f"Per-File MMS-Ergebnis geschrieben nach: {per_file_path}")

        # Für die kombinierte Ausgabe sammeln
        all_rows.extend(rows)

    # Kombinierte CSV schreiben
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(combined_output_path, index=False)
    print(f"Kombinierte MMS-Ergebnisse geschrieben nach: {combined_output_path}")

    # Visualisierung
    try:
        print("Erstelle Visualisierung der MMS-Verteilung...")
        fig_path = output_dir / f"{Path(args.arguments).stem}_mms_distribution.png"
        plt.figure(figsize=(8, 5))
        values = df_all["MMS"].astype(float)
        values = values[~values.isna()]
        plt.hist(values, bins=20)
        plt.xlabel("MMS (Mean of Max Similarities)")
        plt.ylabel("Häufigkeit")
        plt.title("Verteilung der MMS-Ergebnisse (Encoder: bi)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"Visualisierung gespeichert unter: {fig_path}")
    except Exception as e:
        print(f"Konnte Visualisierung nicht erstellen: {e}")


if __name__ == "__main__":
    main()
