from ..data.observable_columns import MODEL_FEATURE_COLUMNS
from ..data.privileged_columns import PRIVILEGED_COLUMNS


def assert_no_privileged_features(features=MODEL_FEATURE_COLUMNS):
    overlap = set(features) & set(PRIVILEGED_COLUMNS)
    if overlap: raise AssertionError(f"Privileged model features: {sorted(overlap)}")


def assert_disjoint_splits(splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap: raise AssertionError(f"Episodes overlap in {left}/{right}: {sorted(overlap)}")

