import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame, target_col: str):
    """
    Performs basic preprocessing:
    - Drops identifier columns
    - Converts target to binary (if needed)
    - Separates features and target
    - One-hot encodes categorical variables

    Scaling is intentionally NOT applied here
    to avoid data leakage. It is handled in train.py.
    """

    # Drop identifier columns
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert target to binary if it's categorical (Yes/No)
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical feature columns
    categorical_cols = X.select_dtypes(include="object").columns

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Return an UNFITTED scaler (fit happens after train-test split)
    scaler = StandardScaler()

    return X, y, scaler
