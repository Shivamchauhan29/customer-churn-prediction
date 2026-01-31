import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple, interpretable engineered features
    specific to the Telco churn dataset.
    """

    X = X.copy()

    # Monthly charges normalized by tenure
    if "MonthlyCharges" in X.columns and "tenure" in X.columns:
        X["charges_per_tenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)

    # Long-term customer indicator (1 year threshold)
    if "tenure" in X.columns:
        X["is_long_term_customer"] = (X["tenure"] >= 12).astype(int)

    return X
