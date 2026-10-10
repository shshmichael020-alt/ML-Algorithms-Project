"""Reproducible multiclass benchmark for CICIDS-style CSV files.

This is a classification benchmark for categorical attack/traffic labels. It does not
turn regression estimates into class IDs. See research/README.md for methodology,
commands, expected outputs, plots, and research reporting requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]
MODEL_NAMES = [
    "Logistic Regression (L2)",
    "Logistic Regression (L1)",
    "Logistic Regression (Elastic Net)",
    "Ridge Classifier",
    "SGD Classifier (log-loss, L2)",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path, files: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    file_metadata = []
    for filename in files:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"CSV not found: {path}. Set --data-dir to the folder containing your files."
            )
        print(f"Reading {path} ...")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        frames.append(frame)
        file_metadata.append({
            "filename": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        })
    data = pd.concat(frames, ignore_index=True, sort=False)
    data.columns = data.columns.astype(str).str.strip()
    return data, file_metadata


def prepare_data(df: pd.DataFrame, target: str):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if target not in df.columns:
        raise ValueError(
            f"Target column {target!r} not found. "
            f"Last columns available: {list(df.columns[-10:])}"
        )

    labels = df[target].astype("string").str.strip()
    valid_target = labels.notna() & labels.ne("")
    dropped_target_rows = int((~valid_target).sum())
    df = df.loc[valid_target].copy()
    labels = labels.loc[valid_target].astype(str)

    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Feature selection based on missingness is deliberately postponed until
    # after splitting, so the held-out test feature distribution is not inspected.
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return X, y, encoder, dropped_target_rows


def make_pipeline(name: str, seed: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if name == "Logistic Regression (L2)":
        model = LogisticRegression(
            solver="lbfgs", max_iter=5000,
            random_state=seed, class_weight="balanced",
        )
    elif name == "Logistic Regression (L1)":
        model = LogisticRegression(
            l1_ratio=1.0, solver="saga", max_iter=5000,
            random_state=seed, class_weight="balanced",
        )
    elif name == "Logistic Regression (Elastic Net)":
        model = LogisticRegression(
            l1_ratio=0.5, solver="saga", max_iter=5000,
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


def safe_filename(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def save_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], model_name: str,
    test_size: float, output_path: Path, normalized: bool = False,
) -> None:
    # Label names can be numerous and long; choose a proportional figure height.
    size = max(8, min(20, 0.48 * len(class_names) + 4))
    if normalized:
        row_totals = matrix.sum(axis=1, keepdims=True)
        plot_matrix = np.divide(
            matrix.astype(float), row_totals,
            out=np.zeros_like(matrix, dtype=float), where=row_totals != 0,
        )
        colorbar_label = "Proportion within true class"
        title_prefix = "Row-normalized Confusion Matrix"
    else:
        plot_matrix = matrix
        colorbar_label = "Number of flows"
        title_prefix = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(size, size * 0.82), constrained_layout=True)
    image = ax.imshow(plot_matrix, interpolation="nearest", aspect="auto",
                      vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, shrink=0.8, label=colorbar_label)
    ax.set(
        title=f"{title_prefix} — {model_name} (test={test_size:.0%})",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
    )
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    if len(class_names) <= 15:
        threshold = plot_matrix.max() / 2 if plot_matrix.size else 0
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                label = f"{plot_matrix[i, j]:.0%}" if normalized else f"{matrix[i, j]:,}"
                ax.text(
                    j, i, label, ha="center", va="center",
                    fontsize=6, color="white" if plot_matrix[i, j] > threshold else "black",
                )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_plots(summary: pd.DataFrame, output_dir: Path, test_size: float) -> None:
    # Score comparison plot
    metric_columns = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]
    long = summary.melt(
        id_vars="model", value_vars=metric_columns,
        var_name="metric", value_name="score",
    )
    pivot = long.pivot(index="model", columns="metric", values="score")
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title(f"Classification Model Comparison (test={test_size:.0%})")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"model_comparison_test_{test_size:.1f}.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Separate figure makes macro-F1 comparisons easier to read.
    ordered = summary.sort_values("f1_macro", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.barh(ordered["model"], ordered["f1_macro"])
    ax.set_title(f"Macro-F1 by Model (test={test_size:.0%})")
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / f"macro_f1_test_{test_size:.1f}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--target-column", default="Label")
    parser.add_argument("--test-size", type=float, choices=[0.2, 0.4, 0.6], default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("research/results"))
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data, input_file_metadata = load_data(args.data_dir, args.files)
    print(f"Loaded {len(data):,} rows × {len(data.columns):,} columns")

    X, y, encoder, dropped_target_rows = prepare_data(data, args.target_column)
    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        rare_ids = counts[counts < 2].index.to_numpy()
        rare_names = encoder.inverse_transform(rare_ids).tolist()
        raise ValueError(
            "Stratified holdout needs at least two rows per target class. "
            f"Classes with fewer than two rows: {rare_names}. "
            "Do not silently remove them; document a defensible alternative split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    missing_train = sorted(set(y_test) - set(y_train))
    if missing_train:
        raise ValueError(
            "These held-out classes are absent from training: "
            f"{encoder.inverse_transform(missing_train).tolist()}"
        )

    # This is a training-only feature check, not a test-set-driven decision.
    train_empty = X_train.columns[X_train.isna().all()].tolist()
    if train_empty:
        print("Dropping columns entirely missing in training split:", train_empty)
        X_train = X_train.drop(columns=train_empty)
        X_test = X_test.drop(columns=train_empty)
    if X_train.shape[1] == 0:
        raise ValueError("All feature columns are entirely missing in the training split.")

    class_ids = np.arange(len(encoder.classes_))
    class_names = encoder.classes_.tolist()
    train_counts = pd.Series(y_train).value_counts().reindex(class_ids, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(class_ids, fill_value=0)
    train_class_counts = {class_names[i]: int(train_counts.iloc[i]) for i in range(len(class_names))}
    test_class_counts = {class_names[i]: int(test_counts.iloc[i]) for i in range(len(class_names))}
    feature_names = X_train.columns.astype(str).tolist()

    rows = []
    for name in MODEL_NAMES:
        print(f"\nFitting {name} ...")
        pipeline = make_pipeline(name, args.seed)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)

        cm = confusion_matrix(y_test, predicted, labels=class_ids)
        report = classification_report(
            y_test, predicted, labels=class_ids,
            target_names=class_names, output_dict=True, zero_division=0,
        )
        metrics = {
            "model": name,
            "task": "multiclass classification",
            "test_size": args.test_size,
            "seed": args.seed,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "n_classes": int(len(class_names)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "precision_weighted": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        file_stem = safe_filename(name)
        (output_dir / f"{file_stem}_test_{args.test_size:.1f}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        save_confusion_matrix(
            cm, class_names, name, args.test_size,
            output_dir / f"{file_stem}_confusion_matrix_test_{args.test_size:.1f}.png",
        )
        save_confusion_matrix(
            cm, class_names, name, args.test_size,
            output_dir / f"{file_stem}_normalized_confusion_matrix_test_{args.test_size:.1f}.png",
            normalized=True,
        )
        rows.append({key: value for key, value in metrics.items()
                     if key not in {"confusion_matrix", "classification_report"}})

        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
        print(f"Macro F1:          {metrics['f1_macro']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"benchmark_test_{args.test_size:.1f}.csv", index=False)
    save_summary_plots(summary, output_dir, args.test_size)

    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_branch": "main",
        "script": "research/run_benchmark.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "input_files": input_file_metadata,
        "target_column": args.target_column,
        "target_missing_or_blank_rows_dropped": dropped_target_rows,
        "input_rows_after_target_filter": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_seed": args.seed,
        "class_names": class_names,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "feature_names_used": feature_names,
        "features_dropped_all_missing_in_training": train_empty,
        "preprocessing": {
            "numeric_conversion": "pd.to_numeric(errors='coerce')",
            "infinity_handling": "positive and negative infinity converted to missing",
            "imputation": "median; fitted only on training partition inside Pipeline",
            "scaling": "StandardScaler; fitted only on training partition inside Pipeline",
            "empty_feature_selection": "columns entirely missing in training partition are removed from both partitions",
        },
        "models": MODEL_NAMES,
        "limitations": [
            "Random stratified split may leak capture/day/campaign similarities across partitions.",
            "No hyperparameter search or cross-validation is performed by this script.",
            "Review features for timestamp, flow identifier, or direct target proxies.",
            "This script evaluates classifiers; its results are not results for the original regression estimators.",
        ],
    }
    (output_dir / f"metadata_test_{args.test_size:.1f}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved CSV, JSON reports, confusion-matrix images, and comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
