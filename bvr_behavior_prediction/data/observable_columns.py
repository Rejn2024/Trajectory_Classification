from .schema import RELATIVE_COLUMNS

# Deliberate allow-list: identity, absolute state, policy and action columns are excluded.
MODEL_FEATURE_COLUMNS = tuple(c for c in RELATIVE_COLUMNS if c != "los_rate") + (
    "track_valid", "track_age",
)

