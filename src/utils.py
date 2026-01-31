import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    """
    Plots confusion matrix from true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
    """

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")

    plt.title(title)
    plt.show()
