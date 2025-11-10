#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Berechnet den RedundancyFree (RF)-Wert und erzeugt Visualisierungen über mehrere
Keypoint-CSV-Dateien hinweg.

Annahmen zu den CSVs:
- eine Spalte mit Keypoint-Texten (Standard: --text-column key_point)
- optional eine Spalte mit Topic (Standard: --topic-column topic). Falls nicht vorhanden,
  wird alles als ein Topic "GLOBAL" behandelt.

Beispielaufrufe:

    # 5 CSV-Dateien, gruppierte Balken + Heatmap + Zusammenfassung als CSV
    python rf.py --csv a.csv b.csv c.csv d.csv e.csv \
        --text-column key_point --topic-column topic \
        --sbert-model ../../Alshomary2021/models/roberta-large-final-model-fold-4-2023-07-05_16-02-50 \
        --out-prefix rf_results

Ausgabe:
- rf_results_summary.csv  : Tabelle (file, topic, n_items, rf)
- rf_results_bars.png     : Gruppierte Balkengrafik (Topics x Dateien)
- rf_results_heatmap.png  : Heatmap (Topics x Dateien)
- rf_results_distribution.png : Boxplot der RF-Verteilung über alle Topics (pro Datei)
- Konsolenausgabe der Top-Zeilen

Hinweis zu Embeddings:
RF basiert auf Ähnlichkeiten zwischen Embeddings (numerische Vektoren),
die semantische Bedeutung von Texten erfassen. Wenn du keine Embeddings
vorliegen hast, erzeugt dieses Skript sie automatisch mit Sentence-Transformers
(--sbert-model).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import numpy as np
import pandas as pd


# Optional: Cross-Encoder für präzisere Paar-Ähnlichkeit (SRR@tau)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # wird zur Laufzeit geprüft


# --- Verbose Logging ---------------------------------------------------------
VERBOSE: bool = False


def vlog(*args: Any, **kwargs: Any) -> None:
    """Print only when --verbose is set."""
    if VERBOSE:
        print(*args, **kwargs)


def ensure_cross_encoder() -> None:
    if CrossEncoder is None:
        raise SystemExit(
            "Fehlende Abhängigkeit: sentence-transformers (CrossEncoder). Installiere mit:\n"
            "  pip install sentence-transformers"
        )


# ------------------------------------------------------------
# Cross-Encoder basierte SRR@tau (Semantic Redundancy Rate)
# ------------------------------------------------------------


def _pair_indices(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def crossencoder_srr(
    texts: List[str], ce_model: "CrossEncoder", tau: float = 0.8
) -> float:
    """Berechne SRR@tau: Anteil der sehr ähnlichen Paare (>= tau) via Cross-Encoder.

    Rückgabe in [0,1], höher = mehr Redundanz. Für n<2 -> 0.0.
    """
    n = len(texts)
    if n < 2:
        return 0.0
    pairs = _pair_indices(n)
    pair_texts = [(texts[i], texts[j]) for (i, j) in pairs]
    scores = ce_model.predict(pair_texts)  # Scores in [0,1]
    # Anteil der Paare >= tau
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    srr = float((scores >= tau).mean())
    vlog(f"[ce] vergleiche {len(pairs)} Paare (n={n}), tau={tau}")
    vlog(f"[ce] SRR@{tau:.2f} = {srr:.4f}")
    return srr


# ------------------------------------------------------------
# CSV -> RF je (file, topic)
# ------------------------------------------------------------


def rf_per_topic_from_csv(
    csv_path: Path,
    text_col: str,
    topic_col: str,
    ce_model: "CrossEncoder",
    ce_tau: float = 0.8,
) -> List[Tuple[str, str, int, float, float]]:
    """Liest eine CSV und gibt eine Liste von (file, topic, n_items, srr_ce, div_ce) zurück."""
    if not csv_path.exists():
        raise SystemExit(f"CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    vlog(
        f"[csv] geladen: {csv_path} mit {len(df)} Zeilen und Spalten {list(df.columns)}"
    )

    if text_col not in df.columns:
        raise SystemExit(
            f"Spalte '{text_col}' nicht in CSV gefunden. Vorhandene Spalten: {list(df.columns)}"
        )

    if topic_col not in df.columns:
        df = df.copy()
        df[topic_col] = "GLOBAL"

    rows: List[Tuple[str, str, int, float, float]] = []
    for topic, g in df.groupby(topic_col):
        texts = g[text_col].astype(str).tolist()
        vlog(f"[topic] '{topic}' mit {len(texts)} Keypoints")
        if len(texts) == 0:
            rows.append((csv_path.name, str(topic), 0, 0.0, 1.0))
            continue
        try:
            srr = crossencoder_srr(texts, ce_model, tau=ce_tau)
        except Exception:
            srr = float("nan")
        div_ce = float(1.0 - srr) if np.isfinite(srr) else float("nan")
        rows.append((csv_path.name, str(topic), len(texts), float(srr), float(div_ce)))
    return rows


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Berechnet die Cross-Encoder-basierte Semantic Redundancy Rate (SRR) und Diversität (div_ce) für mehrere Keypoint-CSV-Dateien."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="Eine oder mehrere CSV-Dateien mit Keypoints.",
    )
    p.add_argument(
        "--text-column",
        default="key_point",
        help="Spaltenname der Texte in den CSVs (Standard: key_point).",
    )
    p.add_argument(
        "--topic-column",
        default="topic",
        help=(
            "Spaltenname des Topics je Keypoint (Standard: topic). \n"
            "Falls nicht vorhanden, wird ein künstliches Topic 'GLOBAL' verwendet."
        ),
    )
    p.add_argument(
        "--ce-model",
        default="../models/stsb-roberta-large",
        help=(
            "Cross-Encoder Modellname/Pfad (lokal oder HF-Hub) für SRR@tau.\n"
            "Beispiel: cross-encoder/stsb-roberta-large oder /pfad/zum/modell"
        ),
    )
    p.add_argument(
        "--ce-threshold",
        type=float,
        default=0.6,
        help=(
            "Schwellwert tau in [0,1] für SRR@tau (Anteil sehr ähnlicher Paare).\n"
            "Empfehlung: 0.75–0.85 nach Kalibrierung."
        ),
    )
    p.add_argument(
        "--out-prefix",
        default="rf_results",
        help="Prefix für Ausgabedateien (CSV).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Ausführliche Konsolen-Logs aktivieren.",
    )
    p.add_argument(
        "--output",
        default="./output",
        help="Outputfolder für die Ergebnisse.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global VERBOSE
    VERBOSE = bool(getattr(args, "verbose", False))
    if VERBOSE:
        print("[cfg] Verbose aktiv")

    # Cross-Encoder laden
    ce_model = None
    vlog(f"[cfg] Cross-Encoder: {args.ce_model}")
    if args.ce_model:
        ensure_cross_encoder()
        ce_model = CrossEncoder(args.ce_model)
        vlog("[init] Cross-Encoder geladen")
    else:
        raise SystemExit("Cross-Encoder Modell muss angegeben werden (--ce-model).")

    vlog(f"[run] verarbeite {len(args.csv)} CSV-Datei(en)")
    # SRR/Div je (file, topic) berechnen
    records: List[Tuple[str, str, int, float, float]] = []
    for csv_path in args.csv:
        vlog(f"[run] >>> {csv_path}")
        recs = rf_per_topic_from_csv(
            csv_path=csv_path,
            text_col=args.text_column,
            topic_col=args.topic_column,
            ce_model=ce_model,
            ce_tau=args.ce_threshold,
        )
        records.extend(recs)
        vlog(f"[run] fertig: {csv_path}")

    if not records:
        print("Keine Ergebnisse erzeugt.")
        sys.exit(0)

    summary_df = pd.DataFrame(
        records,
        columns=["file", "topic", "n_items", "srr_ce", "div_ce"],
    )
    # only keep relevant columns
    summary_df = summary_df[["file", "topic", "div_ce"]].rename(
        columns={"div_ce": "RF"}
    )

    # Speichern als CSV
    out_csv = Path(f"{args.output}/{args.out_prefix}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    vlog(f"[out] schreibe Summary -> {out_csv}")

    # Kurze Übersicht in der Konsole
    print("\nErgebnis-Zusammenfassung (erste Zeilen):")
    with pd.option_context(
        "display.max_rows", 20, "display.max_columns", 10, "display.width", 120
    ):
        print(summary_df.sort_values(["topic", "file"]).head(20))
    print(f"\nGespeichert: {out_csv}")
    vlog("[done] abgeschlossen")


if __name__ == "__main__":
    main()
