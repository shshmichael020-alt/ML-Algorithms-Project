# Research benchmark: methodology, execution, and outputs

This directory contains the corrected and documented **multiclass classification** benchmark for CICIDS-style network traffic labels. Its main purpose is to provide a reproducible, leakage-aware evaluation path for the research paper. It does not replace the historical scripts silently and does not claim scores until run against the local CSV files.

## What was found in the original scripts

- `Linear Regression/LR_0.2_Code.py` and `LR_0.4_Code.py` instantiate `LogisticRegression`, not `LinearRegression`.
- `Linear Regression/LR_0.6_Code.py` fits both `SGDClassifier` and `LogisticRegression`, but the displayed “SGD CLASSIFIER RESULTS” are calculated from `y_pred_sgd`. Its second classifier is not reported as a separate complete result.
- `Ridge Regression/RR_*.py`, `Lasso Regression/LosR_*.py`, and `Elastic Net/EN_*.py` fit regression estimators to integer-encoded class names, then round and clip the continuous predictions to class IDs.
- The historical scripts impute missing values before train/test splitting, so their imputation statistics use held-out feature rows.

Keep historical outputs labelled preliminary. Do not describe their metrics as verified regression/classification benchmark results until you either rerun a methodologically justified replacement or report them explicitly as exploratory historical runs.

## Methodological changes in the new script

1. **Actual model attribution.** It explicitly compares five classification estimators: Logistic Regression with L2, L1, and Elastic-Net penalties; Ridge Classifier; and SGD Classifier with log-loss/L2. These are all classifiers for categorical targets. They are not the original Linear/Ridge/Lasso/Elastic-Net *regression* estimators.
2. **Train-only preprocessing.** Median imputation and standard scaling live inside each scikit-learn pipeline and are fitted by `fit(X_train, y_train)`. Columns with no observed values in the training partition are dropped using training data only, then removed from the test matrix.
3. **No numeric-distance assumption between labels.** Models predict categorical classes directly. No regression output is rounded or clipped into target IDs.
4. **Class-imbalance reporting.** Accuracy, balanced accuracy, weighted precision/recall/F1, macro-F1, per-class reports, and confusion matrices are produced.
5. **Reproducibility metadata.** The run records input CSV names/sizes/SHA-256 hashes, software versions, run timestamp, seed, target, class counts per split, used feature names, preprocessing choices, and excluded target rows.
6. **Plots.** The script saves raw-count and row-normalized confusion-matrix PNGs per model, one metrics-comparison bar plot, and one macro-F1 comparison plot.

## Exact files to run

Run **only** the corrected benchmark script for the new results:

```text
research/run_benchmark.py
```

Do not use the old `Linear Regression/LR_*.py`, `Ridge Regression/RR_*.py`, `Lasso Regression/LosR_*.py`, or `Elastic Net/EN_*.py` outputs as the new benchmark. Those are retained for historical reference. The existing `Logistic Regression/LogR_*.py` files are historical too; the corrected script is the consistent comparison pipeline.

## Step-by-step execution (Windows PowerShell)

Update your local clone after these changes have reached GitHub:

```powershell
git switch main
git pull origin main
python -m pip install -r requirements.txt
```

Check that your three local CSV files exist. If stored under `MachineLearningCSV\MachineLearningCVE`, run from the repository root:

```powershell
python research/run_benchmark.py --data-dir "MachineLearningCSV/MachineLearningCVE" --target-column Label --test-size 0.2 --seed 42
```

If all three CSVs are directly in the repository root, use `--data-dir "."` instead.

For optional sensitivity runs with different holdout proportions:

```powershell
python research/run_benchmark.py --data-dir "MachineLearningCSV/MachineLearningCVE" --target-column Label --test-size 0.4 --seed 42
python research/run_benchmark.py --data-dir "MachineLearningCSV/MachineLearningCVE" --target-column Label --test-size 0.6 --seed 42
```

Each run creates/overwrites files for that particular test size. Keep a copy of each completed result directory before rerunning if you need to preserve it. For the paper, choose the primary evaluation design before treating any result as final. Separate 20%, 40%, and 60% holdouts are sensitivity experiments, not a way to select the best-looking test result.

## Expected output files

For test size 0.2, the output directory is `research/results/` unless changed with `--output-dir`.

- `benchmark_test_0.2.csv` — one row per classifier, containing high-level comparison metrics.
- `metadata_test_0.2.json` — dataset hashes, software versions, split counts, features, preprocessing, and limitations.
- `logistic_regression_l2_test_0.2.json`, `logistic_regression_l1_test_0.2.json`, `logistic_regression_elastic_net_test_0.2.json`, `ridge_classifier_test_0.2.json`, `sgd_classifier_log_loss_l2_test_0.2.json` — full model metrics, class report, and confusion matrix.
- `*_confusion_matrix_test_0.2.png` — five per-model raw-count confusion matrices.
- `*_normalized_confusion_matrix_test_0.2.png` — five per-model row-normalized confusion matrices, useful for comparing class-level recall despite class imbalance.
- `model_comparison_test_0.2.png` — grouped comparison of accuracy, balanced accuracy, weighted-F1, and macro-F1.
- `macro_f1_test_0.2.png` — macro-F1 comparison across classifiers.

There should be twelve PNG figures for a successful run: five raw-count confusion matrices, five row-normalized confusion matrices, and two model-comparison figures. Image readability depends on how many label classes the data contains; the numeric confusion matrices also remain in the JSON reports.

## Test the pipeline before using the real dataset

A synthetic test suite and GitHub Actions workflow are included. These tests do **not** substitute for running the actual research data; they check that the script trains all five estimator types, fits preprocessing from training rows only, writes its expected CSV/JSON/PNG artifacts, and records metadata.

To run tests locally:

```powershell
python -m unittest discover -s tests -v
```

A successful GitHub Actions check validates the script against synthetic data only. The real CICIDS data run and the scientific review remain required.

## Research limitations and decisions to document

- The default split is random and stratified; related flows from the same capture/day/campaign can appear in both training and testing. If the research claim is generalization to unseen periods, networks, or attack campaigns, design and report a time/group-based evaluation as well.
- The script does not perform cross-validation or hyperparameter search. Choose/tune any hyperparameters using training data only, and preserve a final test set untouched until evaluation.
- Review candidate features for IDs, timestamps, and direct target proxies based on the dataset schema and research question. Do not remove/add columns simply because doing so improves the score.
- CICIDS-style datasets can have severe class imbalance. Interpret macro-F1, balanced accuracy, class-level recall, and confusion matrices rather than accuracy alone.
- Models may show convergence warnings, especially with L1 and Elastic Net logistic regression. If that occurs, record the warning and investigate training iterations/scaling/configuration before reporting the results.
- The target column is set to `Label`. If your actual CSV uses another target column, verify the file headers and specify the correct one rather than assuming.
- A held-out class absent from training, a class with too few examples for stratification, or missing/invalid labels must be handled with an explicit documented evaluation decision. The script raises an error rather than silently removing rare classes.
- This script is a classification benchmark. Do not claim that it validates Linear Regression, Ridge Regression, Lasso Regression, or Elastic Net Regression on ordinally encoded labels.

## Reporting checklist for the paper

Record the exact dataset source/version, file hashes, selected files, row/column counts, label-cleaning rule, target-column name, class distribution, complete feature list, dropped features and why, software versions, random seed, train/test counts, model configurations, convergence status, all requested metrics, and figure filenames. Report the evaluation protocol before discussing comparative conclusions. State that the dataset was evaluated locally, and use metrics actually generated by the corrected run.
