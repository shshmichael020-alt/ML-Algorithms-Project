# ML-Algorithms-Project

Machine-learning experiments for network-traffic analysis using CICIDS-style CSV files. The repository contains historical coursework implementations plus a separate, corrected research benchmark.

## Research status

**The older scripts and screenshots are exploratory results, not a leakage-safe validated benchmark.** Inspection of the historical files identified three important issues:

1. `Linear Regression/LR_0.2_Code.py` and `LR_0.4_Code.py` call `LogisticRegression`, not `LinearRegression`. The 60% script fits both SGD and Logistic Regression but reports the SGD results under its results heading.
2. Most historical scripts fit an imputer on the complete dataset before creating the train/test split. That lets held-out features influence preprocessing.
3. Ridge, Lasso, and Elastic Net regression scripts predict integer-encoded category IDs and then round/clip predictions. Attack labels are categorical, not naturally ordered numeric values.

Do not present historical output images or spreadsheet values as validated benchmark results without addressing these limitations. The historical artifacts are retained for traceability.

## Corrected research benchmark

Use these files for the new classification experiment:

- [Research methodology, run instructions, outputs, and reporting checklist](research/README.md)
- [Corrected benchmark script](research/run_benchmark.py)
- [Synthetic end-to-end tests](tests/test_research_benchmark.py)
- [Automated GitHub Actions workflow](.github/workflows/research-benchmark-tests.yml)

The pipeline compares five explicitly named **classifiers**: Logistic Regression with L2, L1, and Elastic-Net regularization; Ridge Classifier; and SGD Classifier with log-loss/L2. These are alternatives for a categorical classification task; they are not equivalent to validating the five regression algorithms in the historical folders.

The corrected script:
- fits median imputation and standard scaling on training rows only;
- predicts categorical target labels directly without rounding regression outputs;
- reports accuracy, balanced accuracy, weighted precision/recall/F1, macro-F1, per-class metrics, and confusion matrices;
- records dataset file hashes, class counts, feature names, preprocessing decisions, random seed, and software versions;
- creates **12 PNG images** per successful run: five raw confusion matrices, five row-normalized confusion matrices, and two model-comparison plots.

### Run on your local dataset

The input CSV files are not stored in this repository. After pulling the latest `main` branch and installing the dependencies, run the benchmark from the repository root.

PowerShell example if your files are in `MachineLearningCSV/MachineLearningCVE`:

```powershell
git switch main
git pull origin main
python -m pip install -r requirements.txt
python research/run_benchmark.py --data-dir "MachineLearningCSV/MachineLearningCVE" --target-column Label --test-size 0.2 --seed 42
```

If the three CSV files are in the current directory, pass `--data-dir "."` instead. The dataset's actual target column must be verified against the CSV headers.

For commands, exact output names, optional 40%/60% test-size runs, and how to interpret the results, see [research/README.md](research/README.md). The primary experiment should use one preselected evaluation protocol; do not choose the test size or model only because it produces the highest test score.

## Tests and validation

Run the test suite locally with:

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions tests use synthetic data. Passing these tests confirms that the script runs on a controlled example and writes the expected artifacts; it does **not** validate performance on the real CICIDS dataset. The real CSV files must still be run, inspected, and documented before any scores are used in a paper.

## Limitations to consider before publication

- A random stratified split can place related network flows from the same capture/day/campaign into both train and test. If the claim is generalization to unseen days, networks, or attack campaigns, consider an appropriately designed time- or group-based split.
- Tune hyperparameters using training data only and preserve final test data for evaluation.
- Inspect the actual feature list for IDs, timestamps, or target proxies, and document any exclusions.
- Since class imbalance can be substantial, interpret macro-F1, balanced accuracy, per-class recall, and normalized confusion matrices alongside accuracy and weighted-F1.
- Record the dataset source/version, file hashes, exact feature list, label distribution, split design, software versions, and actual output metrics.

No real-dataset metrics are claimed by this repository update. The benchmark must be executed on the local data before its results are considered research evidence.
