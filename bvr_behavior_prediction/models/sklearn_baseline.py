SUMMARY_STATS = ("mean", "std", "min", "max", "first", "last", "delta")


def summarise_window(window):
    import numpy as np
    values = np.asarray(window, dtype=float)
    arrays = (values.mean(0), values.std(0), values.min(0), values.max(0),
              values[0], values[-1], values[-1] - values[0])
    return np.concatenate(arrays)


def random_forest(random_state=42, **kwargs):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(class_weight="balanced", random_state=random_state, **kwargs)

