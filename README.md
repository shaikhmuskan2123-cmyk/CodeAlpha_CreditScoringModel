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
## Screenshots

### Dashboard
![overview]
<img width="1912" height="913" alt="overview 1" src="https://github.com/user-attachments/assets/2bc93705-9187-45d8-b4e4-73c8041f3b1b" />
<img width="1917" height="902" alt="overview 2" src="https://github.com/user-attachments/assets/5ea2e950-ff78-4b4c-b13c-7b5e897bb19f" />
<img width="1911" height="905" alt="overview 3" src="https://github.com/user-attachments/assets/e865b551-4658-4d35-8225-9102e2239376" />

### Credit Risk Prediction
![Prediction]
<img width="1910" height="902" alt="Prediction 1" src="https://github.com/user-attachments/assets/cc3c1d77-4e28-4f25-a19e-d8fbf20affdf" />
<img width="1917" height="913" alt="Prediction 2" src="https://github.com/user-attachments/assets/17190e91-7bef-4f3c-a314-ab97a3f6face" />
<img width="1917" height="907" alt="Prediction 3" src="https://github.com/user-attachments/assets/6d20738e-3e5e-4374-ac2c-0afa69acc73a" />

### Dataset Explorer
![Dataset Explorer]
<img width="1917" height="902" alt="Dataset 1" src="https://github.com/user-attachments/assets/d611fa25-0124-4547-9aba-55c044997a85" />
<img width="1917" height="912" alt="Dataset 2" src="https://github.com/user-attachments/assets/5934c913-1702-40d5-b6f1-858588b30f59" />

### Model Performance
![Model Performance]
<img width="1917" height="902" alt="Performance 1" src="https://github.com/user-attachments/assets/256855ff-b990-47d3-8599-429eef4299b1" />
<img width="1917" height="907" alt="Performance 2" src="https://github.com/user-attachments/assets/8d8cb401-22f9-48cf-aa99-d7b256c4039f" />
<img width="1915" height="902" alt="Performance 3" src="https://github.com/user-attachments/assets/1cd37753-4f28-4d0b-b7ae-aa46c804ed3d" />
<img width="1917" height="906" alt="Performance 4" src="https://github.com/user-attachments/assets/1e7733ea-513b-4af7-b334-6bb5908397dd" />

### Batch Prediction
![Batch Prediction]
<img width="1912" height="903" alt="Batch scoring" src="https://github.com/user-attachments/assets/24bfdfa1-a858-427b-95f1-206d6c4afb44" />

## Responsible use
![Responsible use]
<img width="1912" height="907" alt="Responsible" src="https://github.com/user-attachments/assets/f8e749e5-bc2f-4cbb-9158-21563ff8ed7b" />


## Data, limitations, and responsible use

The source is the [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statloggermancreditdata). It is a small historical dataset and may encode outdated or inequitable lending patterns. Its population and time period do not establish that predictions generalize to current applicants. Classification thresholds encode consequential trade-offs: false positives and false negatives have different impacts. No fairness audit, calibration assessment, causal analysis, regulatory review, or independent validation is implied by this educational project. Any real-world use would require representative data, documented governance, careful fairness and compliance review, independent validation, and qualified human oversight.
