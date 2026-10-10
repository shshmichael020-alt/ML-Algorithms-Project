import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Override ML_DATA_DIR when the CSV files are stored elsewhere.
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", r"D:\Python Class\Python Project"))
CSV_FILES = (
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
)

missing_files = [name for name in CSV_FILES if not (DATA_DIR / name).is_file()]
if missing_files:
    missing_list = "\n".join(f"  - {name}" for name in missing_files)
    raise FileNotFoundError(
        f"Missing dataset file(s) in {DATA_DIR}:\n{missing_list}\n"
        "Set the ML_DATA_DIR environment variable to the folder containing the CSV files."
    )

dataset = pd.concat(
    [pd.read_csv(DATA_DIR / name, low_memory=False) for name in CSV_FILES],
    ignore_index=True,
)
dataset.columns = dataset.columns.str.strip()

if "Label" not in dataset.columns:
    raise KeyError("The dataset must contain a 'Label' target column.")

X = dataset.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
y = LabelEncoder().fit_transform(dataset["Label"].astype(str).str.strip())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y,
)

# Fit imputation and scaling only on the training split to prevent leakage.
model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
    ]
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print(
    "\nPrecision : {:.4f}".format(
        precision_score(y_test, y_pred, average="weighted", zero_division=0)
    )
)
print("\nRecall : {:.4f}".format(recall_score(y_test, y_pred, average="weighted")))
print("\nF1 Score : {:.4f}".format(f1_score(y_test, y_pred, average="weighted")))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
