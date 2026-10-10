"""Leakage-safe multiclass benchmark for CICIDS-style CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]


def load_data(data_dir: Path, files: list[str]) -> pd.DataFrame:
    frames = []
    for filename in files:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"CSV not found: {path}. Set --data-dir to the folder containing your files."
            )
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_data(df: pd.DataFrame, target: str):
    df.columns = df.columns.astype(str).str.strip()
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found. Last columns: {list(df.columns[-10:])}")

    labels = df[target].astype("string").str.strip()
    valid = labels.notna() & labels.ne("")
    df = df.loc[valid].copy()
    labels = labels.loc[valid].astype(str)

    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Do not choose/drop columns using the complete dataset. Missingness is
    # evaluated on the training split below to avoid even unsupervised test
    # information influencing feature selection.
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns remain.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return X, y, encoder


def make_pipeline(name: str, seed: int) -> Pipeline:
    # These transformers are fitted only when pipe.fit(X_train, y_train) runs.
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if name == "Logistic Regression (L2)":
        model = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000,
            random_state=seed, class_weight="balanced",
        )
    elif name == "Logistic Regression (L1)":
        model = LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000,
            random_state=seed, class_weight="balanced",
        )
    elif name == "Logistic Regression (Elastic Net)":
        model = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=5000,
            random_state=seed, class_weight="balanced",
        )
    elif name == "Ridge Classifier":
        model = RidgeClassifier(class_weight="balanced")
    elif name == "SGD Classifier (log-loss, L2)":
        model = SGDClassifier(
            loss="log_loss", penalty="l2", max_iter=3000, tol=1e-3,
            random_state=seed, class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--target-column", default="Label")
    parser.add_argument("--test-size", type=float, choices=[0.2, 0.4, 0.6], default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("research/results"))
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    args = parser.parse_args()

    df = load_data(args.data_dir, args.files)
    print(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
    X, y, encoder = prepare_data(df, args.target_column)
    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        rare_ids = counts[counts < 2].index.to_numpy()
        raise ValueError(
            "Stratified holdout needs at least two rows per target class. "
            f"Rare classes: {encoder.inverse_transform(rare_ids).tolist()}. "
            "Choose and document a defensible split/grouping strategy; do not silently drop them."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    missing_train = sorted(set(y_test) - set(y_train))
    if missing_train:
        raise ValueError(f"Classes missing from training: {encoder.inverse_transform(missing_train).tolist()}")

    # Remove features that have no observed values in TRAINING. The same columns
    # are removed from test data, even if test contains values for them.
    train_empty = X_train.columns[X_train.isna().all()].tolist()
    if train_empty:
        print("Dropping columns entirely missing in training split:", train_empty)
        X_train = X_train.drop(columns=train_empty)
        X_test = X_test.drop(columns=train_empty)
    if X_train.shape[1] == 0:
        raise ValueError("All feature columns are entirely missing in the training split.")

    class_ids = np.arange(len(encoder.classes_))
    model_names = [
        "Logistic Regression (L2)",
        "Logistic Regression (L1)",
        "Logistic Regression (Elastic Net)",
        "Ridge Classifier",
        "SGD Classifier (log-loss, L2)",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in model_names:
        pipeline = make_pipeline(name, args.seed)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        report = classification_report(
            y_test, predicted, labels=class_ids, target_names=encoder.classes_,
            output_dict=True, zero_division=0,
        )
        metrics = {
            "model": name,
            "task": "multiclass classification",
            "test_size": args.test_size,
            "seed": args.seed,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(encoder.classes_)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "precision_weighted": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=class_ids).tolist(),
            "classification_report": report,
        }
        safe = name.lower().replace(" ", "_")
        (args.output_dir / f"{safe}_test_{args.test_size:.1f}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        rows.append({k: v for k, v in metrics.items() if k not in {"confusion_matrix", "classification_report"}})

        print(f"\n=== {name} | test_size={args.test_size:.1f} ===")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
        print(f"Macro F1:           {metrics['f1_macro']:.4f}")
        print("\nConfusion matrix (true class IDs × predicted class IDs):")
        print(np.asarray(metrics["confusion_matrix"]))
        print("\nClassification report:")
        print(classification_report(
            y_test, predicted, labels=class_ids, target_names=encoder.classes_,
            zero_division=0,
        ))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / f"benchmark_test_{args.test_size:.1f}.csv", index=False)
    metadata = {
        "files": args.files, "data_dir": str(args.data_dir),
        "target_column": args.target_column, "test_size": args.test_size,
        "seed": args.seed, "class_names": encoder.classes_.tolist(),
        "dropped_features_all_missing_in_training": train_empty,
        "preprocessing": "median imputer and optional scaler fitted inside training-only pipelines",
        "note": "Results must be generated locally because dataset CSVs are not stored in the repository.",
    }
    (args.output_dir / f"metadata_test_{args.test_size:.1f}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved results in: {args.output_dir}")


if __name__ == "__main__":
    main()
