from ..models.gru import GRUEncoder
from ..models.multitask import ACTION_HEADS, MultiTaskModel


def build_model(input_dim, hidden_dim=128):
    return MultiTaskModel(GRUEncoder(input_dim, hidden_dim), hidden_dim, ACTION_HEADS)

