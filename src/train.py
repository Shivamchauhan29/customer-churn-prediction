import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.models import get_models



DATA_PATH = "data/raw/customer_churn.csv"
TARGET = "Churn"

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"


def train_pipeline(test_size: float = 0.2, random_state: int = 42):
    """
    Trains churn prediction models and saves artifacts.

    Returns
    -------
    models : dict
        Trained models
    X_test : pd.DataFrame
        Unscaled test features (for tree-based models)
    X_test_scaled : np.ndarray
        Scaled test features (for logistic regression)
    y_test : pd.Series
        Test labels
    """

    # Create artifact directories safely
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    X, y, scaler = preprocess_data(df, TARGET)
    X = add_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale features for Logistic Regression
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get models
    models = get_models(random_state=random_state)

    # Train and save models
    for name, model in models.items():
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)

        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    return models, X_test, X_test_scaled, y_test


if __name__ == "__main__":
    train_pipeline()
