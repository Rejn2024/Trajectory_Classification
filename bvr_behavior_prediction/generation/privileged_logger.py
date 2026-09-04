from ..data.privileged_columns import PRIVILEGED_COLUMNS


class PrivilegedLogger:
    def __init__(self): self.rows: list[dict] = []
    def append(self, observable: dict, privileged: dict) -> None:
        unknown = set(privileged) - set(PRIVILEGED_COLUMNS)
        if unknown: raise ValueError(f"Undeclared privileged columns: {sorted(unknown)}")
        overlap = set(observable) & set(privileged)
        if overlap: raise ValueError(f"Observable/privileged collision: {sorted(overlap)}")
        self.rows.append({**observable, **privileged})

