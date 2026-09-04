from .observable_columns import MODEL_FEATURE_COLUMNS
from .windows import build_window_indices


class TrajectoryWindowDataset:
    """Framework-neutral indexed dataset; a torch DataLoader can consume it directly."""
    def __init__(self, rows_by_episode, history=25, horizon=0, feature_columns=MODEL_FEATURE_COLUMNS,
                 target_columns=("target_skill_id",)):
        self.rows = rows_by_episode
        self.features = tuple(feature_columns); self.targets = tuple(target_columns)
        self.index = build_window_indices(rows_by_episode, history, horizon); self.history = history

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        item = self.index[index]; rows = self.rows[item.episode_id]
        sequence = [[float(row[c]) for c in self.features]
                    for row in rows[item.end - self.history + 1:item.end + 1]]
        targets = {c: rows[item.target][c] for c in self.targets}
        return sequence, targets

