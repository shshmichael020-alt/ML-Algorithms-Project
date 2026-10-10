# Research benchmark repair

This folder adds a leakage-safe, reproducible multiclass classification benchmark for the CICIDS-style network traffic data used by this project.

## Three issues addressed

1. **Model attribution:** The historical files `Linear Regression/LR_0.2_Code.py` and `LR_0.4_Code.py` actually call `LogisticRegression`. The historical 60% file fits both `SGDClassifier` and `LogisticRegression`. Those numbers should not be presented as verified linear-regression results. The new pipeline emits each model under its actual estimator name.
2. **Preprocessing leakage:** Median imputation and scaling are in a scikit-learn `Pipeline`. They are fitted on training rows only; the test rows are transformed with training-fitted values. Features with no observations in the training split are removed based on training rows only, then removed from the test matrix as well.
3. **Regression-to-class conversion:** This benchmark predicts categorical labels directly. It does not round or clip outputs from regression estimators into encoded attack labels. `RidgeClassifier` is a classifier, not ridge regression with rounded class IDs.

## Run

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If the CSV files are in the current directory:

```bash
python research/run_benchmark.py --data-dir . --target-column Label --test-size 0.2
python research/run_benchmark.py --data-dir . --target-column Label --test-size 0.4
python research/run_benchmark.py --data-dir . --target-column Label --test-size 0.6
```

If they are inside `MachineLearningCSV/MachineLearningCVE`, use:

```bash
python research/run_benchmark.py --data-dir "MachineLearningCSV/MachineLearningCVE" --target-column Label --test-size 0.2
```

Outputs are saved under `research/results/`: a model-specific JSON file with per-class report/confusion matrix, a summary CSV, and run metadata. The CSV data files are not in the repository, so results must be generated in an environment that has them.

## Research-quality cautions

- The 20%, 40%, and 60% test settings are separate holdout experiments. For a stronger paper, choose one primary test set, tune models only within training data using cross-validation, then evaluate once on the untouched test set.
- CICIDS-style labels can be highly imbalanced. Report macro-F1 and balanced accuracy along with accuracy and weighted-F1; review per-class precision/recall and the confusion matrix.
- A random stratified split can place related flows from the same capture/day/campaign in both partitions. If the claim is generalization to unseen days, campaigns, or networks, also evaluate using a temporal or group-based split.
- Check the actual feature list for timestamps, IDs, or features that may directly encode the target. Exclusion should be based on documented domain reasoning, not whichever choice improves the score.
- Treat existing screenshots and spreadsheet metrics as preliminary until regenerated with corrected code. Do not claim that the new pipeline has been validated until it has run successfully on the actual CSV files.
- Retain old scripts as historical artifacts until replacements are compared and reviewed. Do not silently drop rare classes merely to get a stratified split to run.

## Record for the paper

Record data source/version and file hashes, Python and scikit-learn versions, chosen target column, exact feature list, class counts, split method, random seed, preprocessing steps, hyperparameters, any removed columns, per-class results, and output files. Explain the evaluation design and report failures/exclusions transparently.
