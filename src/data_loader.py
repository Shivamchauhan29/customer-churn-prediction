import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset from the given path.

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file

    Returns
    -------
    pd.DataFrame
    """

    path = Path("/Users/shivam/Documents/projects/customer-churn/data/raw/churn_dataset.csv")

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    return df
