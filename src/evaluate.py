import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def evaluate_models(X_test, X_test_scaled, y_test):
    """
    Evaluates trained churn models and saves comparison metrics.
    Handles edge cases where a class may be absent in predictions.
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    models_info = {
        "logistic_regression": {
            "path": MODELS_DIR / "logistic_regression.pkl",
            "X": X_test_scaled
        },
        "decision_tree": {
            "path": MODELS_DIR / "decision_tree.pkl",
            "X": X_test
        },
        "random_forest": {
            "path": MODELS_DIR / "random_forest.pkl",
            "X": X_test
        }
    }

    results = []
    predictions = {}

    for model_name, info in models_info.items():
        model = joblib.load(info["path"])
        X_input = info["X"]

        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        auc = roc_auc_score(y_test, y_prob)

        # Safely extract churn (positive class = 1)
        churn_metrics = report.get("1", {"recall": 0, "precision": 0, "f1-score": 0})

        results.append({
            "model": model_name,
            "recall_churn": churn_metrics["recall"],
            "precision_churn": churn_metrics["precision"],
            "f1_churn": churn_metrics["f1-score"],
            "roc_auc": auc
        })

        predictions[model_name] = {
            "y_pred": y_pred,
            "y_prob": y_prob
        }

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    return metrics_df, predictions
