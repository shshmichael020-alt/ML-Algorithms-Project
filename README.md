# ML Algorithms Project

## Machine Learning for Network Traffic Analysis

This project compares machine-learning classifiers on network-flow data based on the CICIDS dataset. Each script loads the same three traffic captures, preprocesses the data, trains a model, and prints classification results.

The project currently includes:

- Logistic Regression
- Linear Regression folder using `SGDClassifier` with log-loss
- Ridge Regression folder using `RidgeClassifier`
- Lasso Regression folder using `SGDClassifier` with L1 regularization
- Elastic Net folder using `SGDClassifier` with elastic-net regularization

Although several folders retain their original regression names for project continuity, the updated implementations evaluate categorical traffic labels as a classification problem. They do not convert continuous regression predictions into class IDs.

## Repository Structure

Each algorithm folder contains scripts for three train/test splits:

| Script suffix | Training data | Test data |
| --- | ---: | ---: |
| `0.2` | 80% | 20% |
| `0.4` | 60% | 40% |
| `0.6` | 40% | 60% |

Each folder also contains two PNG output images for each split. The workbook [Output of All ML Algo.xlsx](Output%20of%20All%20ML%20Algo.xlsx) contains the consolidated project results.

```text
Elastic Net/          Elastic-net classifier scripts and PNG outputs
Lasso Regression/     L1-regularized classifier scripts and PNG outputs
Linear Regression/    SGD log-loss classifier scripts and PNG outputs
Logistic Regression/  Logistic classifier scripts and PNG outputs
Ridge Regression/     Ridge classifier scripts and PNG outputs
Output of All ML Algo.xlsx
requirements.txt
```

## Dataset

The scripts expect these CICIDS CSV files:

```text
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

The CSV files must contain a `Label` column. The dataset is not included in this repository because of its size and licensing/distribution constraints.

By default, scripts look for the files in:

```text
D:\Python Class\Python Project
```

For another location, set `ML_DATA_DIR` before running a script.

PowerShell:

```powershell
$env:ML_DATA_DIR = "D:\path\to\CICIDS\files"
```

## Installation

Use Python 3.10 or newer and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The main dependencies are Pandas, NumPy, scikit-learn, and Matplotlib.

## Running the Models

Run a script from the repository root. For example:

```powershell
python ".\Logistic Regression\LogR_0.2_Code.py"
python ".\Ridge Regression\RR_0.2_Code.py"
```

The complete script naming pattern is:

```text
Logistic Regression/LogR_{0.2,0.4,0.6}_Code.py
Linear Regression/LR_{0.2,0.4,0.6}_Code.py
Ridge Regression/RR_{0.2,0.4,0.6}_Code.py
Lasso Regression/LosR_{0.2,0.4,0.6}_Code.py
Elastic Net/EN_{0.2,0.4,0.6}_Code.py
```

Each script prints:

- Confusion matrix
- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1 score
- Classification report

## Preprocessing

The scripts use the following pipeline:

1. Combine the three CSV files.
2. Strip whitespace from column names.
3. Separate the `Label` target column from the features.
4. Convert feature values to numeric values; invalid values become missing values.
5. Replace positive and negative infinity with missing values.
6. Encode target labels with `LabelEncoder`.
7. Split the data with a fixed random seed and stratification.
8. Fit mean imputation and standard scaling on the training data only.
9. Train the selected classifier and evaluate it on the test data.

Keeping imputation and scaling inside the scikit-learn pipeline prevents test-set information from leaking into training.

## Results

The repository includes refreshed PNG outputs and the consolidated [Excel results workbook](Output%20of%20All%20ML%20Algo.xlsx). Results depend on the exact source CSV files, preprocessing settings, train/test split, and installed package versions. The committed outputs should therefore be treated as the results for the dataset and settings used to generate them, not as universal benchmark scores.

## Limitations

- The raw CICIDS CSV files are not included.
- The scripts use a random stratified holdout rather than cross-validation or hyperparameter search.
- Network captures can contain related flows across the train and test partitions, which may make holdout performance optimistic.
- The folders named Linear, Lasso, and Elastic Net contain classification implementations using SGD-based estimators because the target is categorical.

## License and Dataset Notice

This repository is intended for educational and comparative machine-learning work. Check the original CICIDS dataset terms before redistributing the raw data or derived materials.
