# CodeAlpha Credit Scoring Model

An educational credit-risk classification project using the UCI Statlog German Credit dataset. It compares Logistic Regression, Decision Tree, and Random Forest pipelines, saves the best cross-validated model, and provides a Streamlit dashboard for exploration and predictions.

This is not a lending, underwriting, or financial decision system. Do not use its output to approve or deny credit.

## Setup

Use Python 3.10 and run these commands from the project root in Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The UCI dataset is downloaded automatically the first time the application or a training/evaluation script loads it. The source file is retained at `data/raw/german_credit.csv`; its validated schema and mapped target are persisted to `data/processed/german_credit_processed.csv` whenever the project loads the dataset.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe evaluate_model.py
```

Training creates the Logistic Regression, Decision Tree, and Random Forest pipelines. It compares mean stratified 5-fold ROC-AUC using only the training partition, selects the highest-scoring model, and saves that pipeline to `models/credit_model_pipeline.joblib`. Model metadata is saved to `models/model_metadata.json`. Held-out probabilities and feature-importance values for all three candidates are saved in `reports/model_metrics.json` for transparent dashboard comparisons; the held-out split must not be used to select a production model or threshold. Diagnostic plots are written to `reports/plots/`.

To score one JSON record or a CSV batch from the command line:

```powershell
.\.venv\Scripts\python.exe predict.py --input path\to\record.json
.\.venv\Scripts\python.exe predict.py --input path\to\records.csv --batch
```

CSV files are treated as batches; `--batch` is accepted as an explicit flag. Batch CSV input must include all 20 feature columns described by the dataset schema. Do not include the target column. Batch predictions are written to standard output as CSV and include predicted class, bad-credit probability, and risk label.

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard includes dataset exploration, live purpose/age/credit-amount/outcome cohort filters, and charts recalculated from the filtered records. Individual predictions appear after you complete the applicant fields and click **Predict Result**. The one-feature sensitivity chart varies one numeric input while holding the submitted profile fixed; it is model sensitivity, not a causal explanation. Change the applicant fields and click **Predict Result** again to calculate an updated result.

Batch CSV scoring includes a downloadable input template and scored-results download. The Performance page compares all three candidates at the selected probability threshold, switches the ROC and feature-importance charts by model, and lets you choose how many features to display. Its threshold analysis includes precision/recall/F1 curves, current-threshold confusion counts, and false-positive/false-negative totals. Threshold analysis is exploratory; do not select a production cutoff using the final test set.

## Target definition

The source dataset encodes credit risk as `1` (good) and `2` (bad). This project maps those labels to:

| Source label | Meaning | Model target |
| --- | --- | ---: |
| 1 | Good credit | 0 |
| 2 | Bad credit | 1 |

Consequently, predicted class `1` and its probability represent the model's estimated bad-credit class. This mapping is implemented in `src/config.py` and validated when loading the dataset.

## Latest verified run

The latest verified training run selected Random Forest by mean 5-fold training ROC-AUC (`0.7912`). Held-out test results were accuracy `0.7550`, precision `0.6667`, recall `0.3667`, F1 `0.4731`, and ROC-AUC `0.7961` at the default `0.50` cutoff. These are results on one small historical dataset, not a claim of suitability for real credit decisions.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Data, limitations, and responsible use

The source is the [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statloggermancreditdata). It is a small historical dataset and may encode outdated or inequitable lending patterns. Its population and time period do not establish that predictions generalize to current applicants. Classification thresholds encode consequential trade-offs: false positives and false negatives have different impacts. No fairness audit, calibration assessment, causal analysis, regulatory review, or independent validation is implied by this educational project. Any real-world use would require representative data, documented governance, careful fairness and compliance review, independent validation, and qualified human oversight.
