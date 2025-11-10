import pandas as pd
from pathlib import Path
import re
import argparse


# Helper to normalize topic string for merge keys
def _normalize_topic_str(s: str) -> str:
    s = str(s)
    # trim and replace underscores with spaces
    s = s.strip().replace("_", " ")
    # collapse multiple whitespace to single space
    s = re.sub(r"\s+", " ", s)
    # drop a single trailing period
    if s.endswith("."):
        s = s[:-1]
    # normalize case for matching
    return s.casefold()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Beide Spalten werden benötigt, sonst brechen wir früh und eindeutig ab
    if "file" not in df.columns or "topic" not in df.columns:
        raise KeyError("Erwarte Spalten 'file' und 'topic' für die Schlüsselerzeugung.")

    # file_key: nur Dateiname, getrimmt, lower/casefold für robustes Matching
    df["file_key"] = (
        df["file"].astype(str).map(lambda x: Path(x).name).str.strip().str.casefold()
    )

    # topic_key über Normalisierer
    df["topic_key"] = df["topic"].map(_normalize_topic_str)
    return df


def merge_results(args: argparse.Namespace) -> None:
    # Ausgabeverzeichnis sicherstellen
    args.output.mkdir(parents=True, exist_ok=True)

    # CSV-Dateien nach Präfixen
    files = {
        "iqs": args.output / "iqs_all_results.csv",
        "mms": args.output / "mms_all_results.csv",
        "ac": args.output / "ac_all_results.csv",
        "rf": args.output / "rf_all_results.csv",
    }

    # Einlesen
    df_iqs = pd.read_csv(files["iqs"])
    print(f"Loaded IQS CSV: shape={df_iqs.shape}")
    print(df_iqs.head())
    df_mms = pd.read_csv(files["mms"])
    print(f"Loaded MMS CSV: shape={df_mms.shape}")
    print(df_mms.head())
    df_ac = pd.read_csv(files["ac"])
    print(f"Loaded AC CSV: shape={df_ac.shape}")
    print(df_ac.head())
    df_rf = pd.read_csv(files["rf"])
    print(f"Loaded RF CSV: shape={df_rf.shape}")
    print(df_rf.head())

    # Normalisieren der Schlüsselspalten, damit Pfade/Whitespace vereinheitlicht sind
    df_iqs = _standardize_columns(df_iqs)
    print("After standardizing IQS: unique file_keys:", df_iqs["file_key"].unique())
    print("After standardizing IQS: unique topic_keys:", df_iqs["topic_key"].unique())
    df_mms = _standardize_columns(df_mms)
    print("After standardizing MMS: unique file_keys:", df_mms["file_key"].unique())
    print("After standardizing MMS: unique topic_keys:", df_mms["topic_key"].unique())
    df_ac = _standardize_columns(df_ac)
    print("After standardizing AC: unique file_keys:", df_ac["file_key"].unique())
    print("After standardizing AC: unique topic_keys:", df_ac["topic_key"].unique())
    df_rf = _standardize_columns(df_rf)
    print("After standardizing RF: unique file_keys:", df_rf["file_key"].unique())
    print("After standardizing RF: unique topic_keys:", df_rf["topic_key"].unique())

    # Umbenennen der Wertspalten
    df_iqs = df_iqs.rename(columns={"IQS": "iqs"})
    print("After renaming IQS columns:", df_iqs.columns)
    df_mms = df_mms.rename(columns={"MMS": "mms"})
    print("After renaming MMS columns:", df_mms.columns)
    df_ac = df_ac.rename(columns={"AC": "ac"})
    print("After renaming AC columns:", df_ac.columns)
    df_rf = df_rf.rename(columns={"RF": "rf"})
    print("After renaming RF columns:", df_rf.columns)

    # Vor dem Mergen: Duplikate auf RHS-Tabellen entfernen, um kartesische Produkte zu verhindern
    before_rf = len(df_rf)
    df_rf = df_rf.drop_duplicates(subset=["file_key", "topic_key"])
    after_rf = len(df_rf)
    print(f"Deduped RF rows on keys: {before_rf} -> {after_rf}")

    before_iqs = len(df_iqs)
    df_iqs = df_iqs.drop_duplicates(subset=["file_key", "topic_key"])
    after_iqs = len(df_iqs)
    print(f"Deduped IQS rows on keys: {before_iqs} -> {after_iqs}")

    before_mms = len(df_mms)
    df_mms = df_mms.drop_duplicates(subset=["file_key", "topic_key"])
    after_mms = len(df_mms)
    print(f"Deduped MMS rows on keys: {before_mms} -> {after_mms}")

    # Entferne doppelte Identifikatoren aus rechten Tabellen, damit Merge keine
    # file/topic-Kollisionen erzeugt. Wir behalten die AC-Originalspalten.
    def _drop_nonkey_identifiers(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in ["file", "topic"] if c in df.columns])

    df_rf = _drop_nonkey_identifiers(df_rf)
    df_iqs = _drop_nonkey_identifiers(df_iqs)
    df_mms = _drop_nonkey_identifiers(df_mms)

    print("RF columns before merge:", df_rf.columns.tolist())
    print("IQS columns before merge:", df_iqs.columns.tolist())
    print("MMS columns before merge:", df_mms.columns.tolist())

    # Zusammenführen
    print(
        f"Merging AC and RF on keys ['file_key','topic_key']: shapes {df_ac.shape} & {df_rf.shape}"
    )
    merged = df_ac.merge(df_rf, on=["file_key", "topic_key"], validate="one_to_one")
    print(f"After merging AC+RF: shape {merged.shape}")
    print(
        f"Merging previous result with IQS on keys ['file_key','topic_key']: shapes {merged.shape} & {df_iqs.shape}"
    )
    merged = merged.merge(df_iqs, on=["file_key", "topic_key"], validate="one_to_one")
    print(f"After merging with IQS: shape {merged.shape}")
    print(
        f"Merging previous result with MMS on keys ['file_key','topic_key']: shapes {merged.shape} & {df_mms.shape}"
    )
    merged = merged.merge(df_mms, on=["file_key", "topic_key"], validate="one_to_one")
    print(f"After final merge: shape {merged.shape}")
    print(merged.head())
    print("Distinct topic_keys after merge:", merged["topic_key"].nunique())

    # SUSWIR berechnen: 0.25 * (IQS + MMS + RF + AC)
    for c in ["iqs", "mms", "rf", "ac"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        else:
            raise KeyError(
                f"Spalte '{c}' fehlt im zusammengeführten DataFrame – SUSWIR kann nicht berechnet werden."
            )

    merged["suswir"] = 0.25 * (
        merged["iqs"] + merged["mms"] + merged["rf"] + merged["ac"]
    ).astype(float)
    # Optional: runden für schönere Anzeige
    merged["suswir"] = merged["suswir"].round(6)

    print("Added SUSWIR column. Preview:")
    preview_cols = [
        c
        for c in ["file", "topic", "iqs", "mms", "rf", "ac", "suswir"]
        if c in merged.columns
    ]
    print(merged[preview_cols].head())
    print("SUSWIR min/max:", merged["suswir"].min(), merged["suswir"].max())

    # Speichern
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "merged_results.csv"
    print(f"Saving merged results to: {output_file}")
    merged.to_csv(output_file, index=False)
    print(f"Fertig! Ergebnisse gespeichert in {output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fasst die Ergebnisse der reference-free Evaluierung (IQS, MMS, AC, RF) zusammen."
        )
    )
    p.add_argument(
        "--output",
        type=Path,
        default="output",
        required=False,
        help="Pfad zur Ausgabedatei (CSV) für die zusammengefassten Ergebnisse.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_results(args)
