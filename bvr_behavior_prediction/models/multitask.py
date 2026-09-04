from torch import nn

CURRENT_HEADS = {"lateral": 3, "vertical": 3, "energy": 3, "tactical": 10}
ACTION_HEADS = {"heading_action": 15, "altitude_action": 15, "speed_action": 9, "fire_action": 2}


class MultiTaskModel(nn.Module):
    def __init__(self, encoder, latent_dim, heads=None):
        super().__init__(); self.encoder = encoder
        heads = heads or {**CURRENT_HEADS, "transition_2s": 2, "transition_4s": 2,
                          "transition_8s": 2, "next_tactical": 10, **ACTION_HEADS}
        self.heads = nn.ModuleDict({name: nn.Linear(latent_dim, size) for name, size in heads.items()})
    def forward(self, x):
        latent = self.encoder(x); return {name: head(latent) for name, head in self.heads.items()}

