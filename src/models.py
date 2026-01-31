from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_models(random_state: int = 42):
    """
    Returns a dictionary of initialized models used for churn prediction.

    All models:
    - Support probability estimates (predict_proba)
    - Handle class imbalance via class_weight
    """

    models = {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }

    return models
