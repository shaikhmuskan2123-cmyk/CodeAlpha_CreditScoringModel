# Dataset folder

This project uses the UCI Statlog German Credit dataset.

- Source: https://archive.ics.uci.edu/dataset/144/statloggermancreditdata
- Data file: `data/raw/german_credit.csv`
- Validated data: `data/processed/german_credit_processed.csv`
- Raw source: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

The label encoding from the original dataset is mapped to:

- Good credit = 0
- Bad credit = 1

The official file uses `1` for good and `2` for bad, so the project converts that to the required binary target convention.

The validated CSV is generated automatically when the dataset loader runs. The raw download remains unchanged.
