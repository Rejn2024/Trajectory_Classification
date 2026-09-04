from torch import nn


class AircraftEmbedding(nn.Module):
    def __init__(self, num_aircraft_types, embedding_dim=8):
        super().__init__(); self.embedding = nn.Embedding(num_aircraft_types, embedding_dim)
    def forward(self, observer_type, target_type):
        return self.embedding(observer_type), self.embedding(target_type)

