from ..models.sklearn_baseline import random_forest, summarise_window


def fit_random_forest(windows, labels, **kwargs):
    model = random_forest(**kwargs)
    model.fit([summarise_window(window) for window in windows], labels)
    return model

