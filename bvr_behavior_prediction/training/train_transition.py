from ..models.gru import GRUEncoder
from ..models.multitask import MultiTaskModel


def build_model(input_dim, hidden_dim=128):
    return MultiTaskModel(GRUEncoder(input_dim, hidden_dim), hidden_dim,
                          {"transition_2s": 2, "transition_4s": 2, "transition_8s": 2,
                           "next_tactical": 10})

