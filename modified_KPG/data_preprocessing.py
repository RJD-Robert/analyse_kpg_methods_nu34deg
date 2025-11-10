#!/usr/bin/env python3
# Preprocessing: select top-K arguments per topic
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Tuple
import os
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

import time

# Optional Progressbar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Laufzeit-Konfiguration (gefüllt in main())
CFG = {
    "progress_bar": True,
    "save_plots": False,
    "plot_dir": None,
}


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


def _prepare_pairs(
    df: pd.DataFrame,
    arg_col: str,
    topic_col: Optional[str],
    default_topic: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if arg_col not in df.columns:
        raise ValueError(f"Spalte '{arg_col}' nicht in CSV gefunden.")
    if topic_col is None and default_topic is None:
        raise ValueError(
            "Entweder '--topic-col' angeben oder '--default-topic' setzen."
        )

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
    return df, df_pairs


def score_pairs(
    df_pairs: pd.DataFrame,
    model_name: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Scort eindeutige (Argument, Topic)-Paare und mapt die Scores zurück."""
    total_rows = len(df_pairs)
    print(f"Eingelesen: {total_rows} Zeilen.")

    # Deduplizieren nach (Argument, Topic)
    df_unique = df_pairs.drop_duplicates(subset=["___arg", "___topic"]).reset_index(
        drop=True
    )
    print(
        f"Eindeutige (Argument,Topic)-Paare: {len(df_unique)} (Einsparung: {total_rows - len(df_unique)})"
    )

    u_args = df_unique["___arg"].tolist()
    u_topics = df_unique["___topic"].tolist()

    tokenizer, model, device, is_custom = load_model(model_name)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    all_scores: List[float] = []

    # Optionaler Fortschrittsbalken
    use_pbar = bool(CFG.get("progress_bar", True)) and tqdm is not None
    pbar = None
    if use_pbar:
        pbar = tqdm(total=len(u_args), desc="Scoring", unit="arg")

    model.eval()
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

            # Debug: Beispiel-Scores aus dem ersten Batch ausgeben
            if batch_idx == 0:
                try:
                    preview = probs[:5].flatten().tolist()
                except Exception:
                    preview = probs[:5].tolist()
                print("Beispiel-Scores erster Batch:", preview)

            all_scores.extend(probs.tolist())
            if pbar is not None:
                try:
                    pbar.update(len(arg_batch))
                except Exception:
                    pass

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    df_unique_scores = df_unique.copy()
    df_unique_scores["quality_score"] = all_scores

    # Zurück auf alle Zeilen mappen (many-to-one)
    df_scored = df_pairs.merge(
        df_unique_scores,
        on=["___arg", "___topic"],
        how="left",
        validate="many_to_one",
    )

    # Ausführliche Verteilungsanalyse und Statistik
    scores_np = df_scored["quality_score"].to_numpy(dtype=float)
    print("=" * 60)
    print("Verteilungsanalyse der Quality Scores (alle Zeilen):")
    count = np.sum(~np.isnan(scores_np))
    print(f"  Anzahl: {int(count)}")
    min_v = np.nanmin(scores_np)
    max_v = np.nanmax(scores_np)
    mean_v = np.nanmean(scores_np)
    std_v = np.nanstd(scores_np)
    med_v = np.nanmedian(scores_np)
    # Perzentile
    p0, p5, p10, p25, p50, p75, p90, p95, p100 = np.nanpercentile(
        scores_np, [0, 5, 10, 25, 50, 75, 90, 95, 100]
    )
    print(f"  Min:    {min_v:.4f}")
    print(f"  Max:    {max_v:.4f}")
    print(f"  Mittel: {mean_v:.4f}")
    print(f"  Std:    {std_v:.4f}")
    print(f"  Median: {med_v:.4f}")
    print("  Perzentile:")
    print(f"    10%:  {p10:.4f}  |  25%: {p25:.4f}  |  50% (Median): {p50:.4f}")
    print(f"    75%:  {p75:.4f}  |  90%: {p90:.4f}  |   95%: {p95:.4f}")
    print(f"    0%:   {p0:.4f}   | 100%: {p100:.4f}")
    # Bereiche für Nachvollziehbarkeit
    print("  Bereiche:")
    print(f"    Top-10% liegen zwischen {p90:.4f} und {p100:.4f}.")
    print(f"    Zentrale 90% liegen zwischen {p5:.4f} und {p95:.4f}.")
    print(f"    Zentrale 80% liegen zwischen {p10:.4f} und {p90:.4f}.")
    # Histogramm grob (10 Bins)
    try:
        hist_counts, bin_edges = np.histogram(
            scores_np[~np.isnan(scores_np)], bins=10, range=(min_v, max_v)
        )
        print("  Grobes Histogramm (10 Bins):")
        for i in range(len(hist_counts)):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            print(f"    [{left:.3f}, {right:.3f}): {int(hist_counts[i])}")
    except Exception:
        pass
    print("=" * 60)

    # Optional: Plots speichern (Histogramm + CDF)
    if CFG.get("save_plots", False):
        try:
            import matplotlib.pyplot as plt

            # Zielverzeichnis bestimmen
            plot_dir = CFG.get("plot_dir")
            if not plot_dir:
                plot_dir = os.getcwd()
            os.makedirs(plot_dir, exist_ok=True)

            # Nur gültige Scores
            valid = scores_np[~np.isnan(scores_np)]
            if valid.size > 0:
                # Histogramm
                plt.figure()
                plt.hist(valid, bins=30)
                plt.xlabel("quality_score")
                plt.ylabel("Häufigkeit")
                plt.title("Histogramm der Quality Scores")
                hist_path = os.path.join(plot_dir, "quality_score_hist.png")
                plt.savefig(hist_path, bbox_inches="tight")
                plt.close()

                # CDF
                vs = np.sort(valid)
                cdf = np.arange(1, len(vs) + 1) / len(vs)
                plt.figure()
                plt.plot(vs, cdf)
                plt.xlabel("quality_score")
                plt.ylabel("kumulative Verteilung")
                plt.title("CDF der Quality Scores")
                cdf_path = os.path.join(plot_dir, "quality_score_cdf.png")
                plt.savefig(cdf_path, bbox_inches="tight")
                plt.close()

                print(f"Plots gespeichert: {hist_path} und {cdf_path}")
        except Exception as e:
            print(f"Warnung: Konnte Plots nicht speichern: {e}")

    return df_scored


def select_top_k(
    df_input: pd.DataFrame,
    df_scored_pairs: pd.DataFrame,
    arg_col: str,
    topic_col: Optional[str],
    stance_col: Optional[str],
    default_topic: Optional[str],
    top_k: int,
    dedup_on_arg: bool,
    keep_score: bool = False,
) -> pd.DataFrame:
    """Wählt Top-K Argumente **pro Topic** auf Basis des quality_score und gibt optional die Stance-Spalte mit aus."""

    # Join der Scores ans Original-DF über technische Spalten
    if topic_col is not None:
        df_join = pd.DataFrame(
            {
                "___arg": df_input[arg_col].fillna("").astype(str),
                "___topic": df_input[topic_col].fillna("").astype(str),
            }
        )
    else:
        df_join = pd.DataFrame(
            {
                "___arg": df_input[arg_col].fillna("").astype(str),
                "___topic": [default_topic] * len(df_input),
            }
        )

    df_with_scores = df_input.copy()
    df_with_scores = pd.concat([df_with_scores, df_join], axis=1)

    # Sicherstellen, dass die rechten Merge-Keys eindeutig sind
    # (score_pairs() liefert alle Zeilen zurueck; dadurch koennen (___arg, ___topic)
    # mehrfach vorkommen. Wir aggregieren daher auf eindeutige Paare.)
    df_scores_unique = df_scored_pairs.groupby(["___arg", "___topic"], as_index=False)[
        "quality_score"
    ].max()

    df_with_scores = df_with_scores.merge(
        df_scores_unique,
        on=["___arg", "___topic"],
        how="left",
        validate="many_to_one",
    )

    # Falls dedup gewünscht: innerhalb eines Topics je Argument nur den höchsten Score behalten
    if dedup_on_arg:
        subset_cols = [arg_col]
        if topic_col is not None:
            # Sortieren, damit der höchste Score pro (topic, arg) bleibt
            df_with_scores = df_with_scores.sort_values(
                [topic_col, "quality_score"], ascending=[True, False]
            ).drop_duplicates(subset=[topic_col, arg_col], keep="first")
        else:
            df_with_scores = df_with_scores.sort_values(
                "quality_score", ascending=False
            ).drop_duplicates(subset=subset_cols, keep="first")

    # Auswahl: pro Topic Top-K; ohne Topic -> global Top-K
    if topic_col is not None:
        df_top = (
            df_with_scores.sort_values(
                [topic_col, "quality_score"], ascending=[True, False]
            )
            .groupby(topic_col, group_keys=True)
            .head(top_k)
            .reset_index(drop=True)
        )
    else:
        df_top = (
            df_with_scores.sort_values("quality_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    # Technische Spalten entfernen
    for c in ["___arg", "___topic"]:
        if c in df_top.columns:
            del df_top[c]

    # Standardausgabe nur Topic/Argument (+ optional Score)
    base_cols = [col for col in [topic_col, stance_col, arg_col] if col is not None]
    if keep_score:
        out_cols = base_cols + ["quality_score"]
    else:
        out_cols = base_cols

    # Falls kein topic_col genutzt wurde, aber default_topic gesetzt ist, bleibt die Struktur wie im Input
    out_cols = [c for c in out_cols if c in df_top.columns]

    return df_top[out_cols]


def main():
    ap = argparse.ArgumentParser(
        description="Wählt die Top-K qualitativ hochwertigsten Argumente per Qualitätsmodell aus."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Pfad zur Eingabe-CSV",
        default="../data/argmatch/argmatch_arguments.csv",
    )
    ap.add_argument("--arg-col", required=False, help="Spaltenname der Argumente")
    ap.add_argument(
        "--topic-col", required=False, help="Spaltenname für Topics (optional)"
    )
    ap.add_argument(
        "--stance-col", required=False, help="Spaltenname für Stance (optional)"
    )
    ap.add_argument(
        "--default-topic",
        required=False,
        help="Topic für alle Zeilen, falls --topic-col fehlt",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Pfad zur Ausgabe-CSV (Top-K)",
        default="./argmatch_arguments_topk.csv",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=2000,
        help="Wieviele Argumente ausgeben (Default: 2000)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batchgröße fürs Tokenisieren/Inferieren",
    )
    ap.add_argument(
        "--model-name",
        default="../alshomary2021/models/argument-quality-ibm-reproduced/bert_wa",
        help="HF-Checkpoint oder Custom-Webis-Ordner (mit model.pt & training_config.json)",
    )
    ap.add_argument(
        "--dedup-on-arg",
        action="store_true",
        help="Duplikate auf Argument-Text-Ebene vor der Top-K-Auswahl entfernen",
    )
    ap.add_argument(
        "--keep-score",
        action="store_true",
        help="quality_score in der Ausgabe behalten (Default: nur topic/argument)",
    )
    ap.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Deaktiviert den Fortschrittsbalken während des Scorings",
    )
    ap.add_argument(
        "--save-plots",
        action="store_true",
        help="Speichert Histogramm und CDF der Scores als PNG",
    )
    ap.add_argument(
        "--plot-dir",
        default=None,
        help="Zielverzeichnis für gespeicherte Plots (Default: neben --out)",
    )
    args = ap.parse_args()

    # Laufzeitkonfiguration setzen
    CFG["progress_bar"] = not args.no_progress_bar
    CFG["save_plots"] = bool(args.save_plots)
    # Plot-Ziel: Ordner von --out oder explizit
    if args.plot_dir is not None and len(str(args.plot_dir).strip()) > 0:
        CFG["plot_dir"] = args.plot_dir
    else:
        try:
            CFG["plot_dir"] = os.path.dirname(os.path.abspath(args.out))
        except Exception:
            CFG["plot_dir"] = os.getcwd()

    df_in = pd.read_csv(args.csv)
    # Doppelte Argumente entfernen
    if args.arg_col and args.arg_col in df_in.columns:
        before = len(df_in)
        df_in = df_in.drop_duplicates(subset=[args.arg_col]).reset_index(drop=True)
        after = len(df_in)
        print(
            f"Duplikate entfernt: {before - after} Zeilen gelöscht, {after} verbleibend."
        )
    # Debug: Eingangsdaten prüfen
    print(f"Eingabedaten geladen: {df_in.shape[0]} Zeilen, {df_in.shape[1]} Spalten")
    print(f"Spaltennamen: {list(df_in.columns)}")

    # Sanity-Check für die erwarteten Standardspaltennamen
    if (
        args.topic_col is None
        and args.default_topic is None
        and "topic" in df_in.columns
    ):
        # Falls CSV die Spalten topic/argument hat, aber topic_col nicht übergeben wurde, verwenden wir sie automatisch
        args.topic_col = "topic"
    if args.arg_col is None and "argument" in df_in.columns:
        args.arg_col = "argument"
    if args.stance_col is None and "stance" in df_in.columns:
        args.stance_col = "stance"

    # Paare bauen
    df_orig, df_pairs = _prepare_pairs(
        df=df_in,
        arg_col=args.arg_col,
        topic_col=args.topic_col,
        default_topic=args.default_topic,
    )
    # Debug: nach _prepare_pairs
    print(f"Pairs erstellt: {df_pairs.shape}")
    print(df_pairs.head(3))

    # Scoring
    df_scored_pairs = score_pairs(
        df_pairs=df_pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    # Debug: nach score_pairs
    print(f"Scored-Pairs: {df_scored_pairs.shape}")
    print(df_scored_pairs.head(3))

    # Alle Argumente mit Scores mergen und komplett sortiert ausgeben/speichern
    if args.topic_col is not None:
        df_join_all = pd.DataFrame(
            {
                "___arg": df_in[args.arg_col].fillna("").astype(str),
                "___topic": df_in[args.topic_col].fillna("").astype(str),
            }
        )
    else:
        df_join_all = pd.DataFrame(
            {
                "___arg": df_in[args.arg_col].fillna("").astype(str),
                "___topic": [args.default_topic] * len(df_in),
            }
        )

    df_scores_unique_all = df_scored_pairs.groupby(
        ["___arg", "___topic"], as_index=False
    )["quality_score"].max()

    df_all_with_scores = pd.concat([df_in.copy(), df_join_all], axis=1).merge(
        df_scores_unique_all,
        on=["___arg", "___topic"],
        how="left",
        validate="many_to_one",
    )

    # Sortiert nach Score (absteigend)
    df_all_sorted = df_all_with_scores.sort_values(
        "quality_score", ascending=False
    ).reset_index(drop=True)

    # Vorschau drucken und Dateipfad bestimmen
    print("Alle Argumente, nach Qualitätswert sortiert (Top 20 Vorschau):")
    preview_cols = [
        c
        for c in [args.topic_col, args.stance_col, args.arg_col, "quality_score"]
        if c is not None
    ]
    print(df_all_sorted[preview_cols].head(20))

    # Ausgabedatei für komplette sortierte Liste bestimmen
    root, ext = os.path.splitext(args.out)
    if ext == "":
        ext = ".csv"
    out_all = f"{root}_ALL_sorted{ext}"
    df_all_sorted.drop(
        columns=[c for c in ["___arg", "___topic"] if c in df_all_sorted.columns],
        inplace=True,
    )
    df_all_sorted.to_csv(out_all, index=False)
    print(
        f"Komplette, absteigend sortierte Liste aller Argumente nach Qualitätswert gespeichert unter: {out_all}"
    )

    # Auswahl Top-K
    df_top = select_top_k(
        df_input=df_orig,
        df_scored_pairs=df_scored_pairs,
        arg_col=args.arg_col,
        topic_col=args.topic_col,
        stance_col=args.stance_col,
        default_topic=args.default_topic,
        top_k=args.top_k,
        dedup_on_arg=args.dedup_on_arg,
        keep_score=args.keep_score,
    )
    # Debug: nach select_top_k
    print(f"Top-K Auswahl fertig: {df_top.shape}")
    print(df_top.head(10))

    # Debug: vor dem Schreiben der Datei
    print(f"Schreibe Ergebnis nach: {args.out}")
    # Schreiben
    df_top.to_csv(args.out, index=False)
    print(
        f"Fertig. Pro Topic wurden bis zu Top-{args.top_k} Argumente nach {args.out} geschrieben."
    )


if __name__ == "__main__":
    print("Starte Datenvorverarbeitung für Top-K Argumentauswahl...")
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Gesamtdauer: {elapsed:.2f} Sekunden")
