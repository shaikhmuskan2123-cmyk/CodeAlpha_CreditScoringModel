import pandas as pd

from src.preprocessing import build_preprocessor, split_train_test


def test_preprocessor_fit_transform_and_split():
    data = {
        "num_a": [1, 2, 3, 4, 5, 6],
        "num_b": [10, 11, 12, 13, 14, 15],
        "cat_x": ["A", "B", "A", "B", "A", "B"],
        "credit_risk": [0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = split_train_test(df, target_column="credit_risk", test_size=0.33, random_state=42)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    assert X_train_proc.shape[0] == len(X_train)
    assert X_test_proc.shape[0] == len(X_test)
    assert X_train_proc.shape[1] > 0
    assert y_train.shape[0] == len(X_train)
    assert y_test.shape[0] == len(X_test)
