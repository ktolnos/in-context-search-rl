import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input: int, hidden_dim: int, output: int, num_layers: int = 2):
        super(MLPModel, self).__init__()
        self.input = input
        self.hidden_dim = hidden_dim
        self.output = output
        self.num_layers = num_layers
        self.input_fc = nn.Linear(input, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_fc = nn.Linear(hidden_dim, output)
        self.activation = nn.ReLU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self._init_weights()

    def _init_weights(self):
        for layer in self.fcs:
            nn.init.orthogonal_(layer.weight, nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.output_fc.weight, 1.0)
        nn.init.zeros_(self.output_fc.bias)

    def forward(self, x):
        x = self.activation(self.input_fc(x))
        for fc, layer_norm in zip(self.fcs, self.layer_norms):
            x = self.activation(layer_norm(fc(x)))
        return self.output_fc(x)


class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = MLPModel(obs_shape, hidden_dim, latent_dim)

    def forward(self, x):
        return self.model(x)


class InvModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_shape):
        super(InvModel, self).__init__()
        self.model = MLPModel(2 * latent_dim, hidden_dim, action_shape)

    def forward(self, x):
        return self.model(x)


class ForwardModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_shape):
        super(ForwardModel, self).__init__()
        self.model = MLPModel(latent_dim + action_shape, hidden_dim, latent_dim)

    def forward(self, obs, action):
        return self.model(torch.cat([obs, action], dim=-1))


class RewardModel(nn.Module):
    def __init__(self, latent_dim, action_shape, hidden_dim):
        super(RewardModel, self).__init__()
        self.model = MLPModel(latent_dim * 2 + action_shape, hidden_dim, 1)

    def forward(self, z, a, z_next):
        return self.model(torch.cat([z, a, z_next], dim=-1))