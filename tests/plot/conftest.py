import matplotlib
import pytest

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")


@pytest.fixture
def simple_data():
    """Simple 3-class data with some misclassifications."""
    y_true = ["fall", "fall", "walk", "walk", "sitting", "sitting"]
    y_pred = ["fall", "walk", "walk", "walk", "sitting", "fall"]
    return y_true, y_pred


@pytest.fixture
def multiclass_data():
    """Larger multi-class data for subset tests."""
    y_true = [
        "fall",
        "fall",
        "fallen",
        "walk",
        "walk",
        "sitting",
        "standing",
        "lying",
        "crawl",
        "jump",
    ]
    y_pred = [
        "fall",
        "walk",
        "fallen",
        "walk",
        "sitting",
        "sitting",
        "standing",
        "lying",
        "crawl",
        "walk",
    ]
    return y_true, y_pred
