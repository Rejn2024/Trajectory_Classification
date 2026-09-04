from ..models.gru import GRUEncoder
from ..models.multitask import CURRENT_HEADS, MultiTaskModel


def build_model(input_dim, hidden_dim=128):
    return MultiTaskModel(GRUEncoder(input_dim, hidden_dim), hidden_dim, CURRENT_HEADS)

