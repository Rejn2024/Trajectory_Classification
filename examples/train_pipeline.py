"""Minimal wiring example after trajectories have been generated."""
from bvr_behavior_prediction.data.observable_columns import MODEL_FEATURE_COLUMNS
from bvr_behavior_prediction.data.torch_dataset import TrajectoryWindowDataset
from bvr_behavior_prediction.training.train_current_state import build_model


def prepare(rows_by_episode):
    dataset = TrajectoryWindowDataset(rows_by_episode, history=25,
                                      target_columns=("lateral", "vertical", "energy", "tactical"))
    model = build_model(len(MODEL_FEATURE_COLUMNS))
    return dataset, model

