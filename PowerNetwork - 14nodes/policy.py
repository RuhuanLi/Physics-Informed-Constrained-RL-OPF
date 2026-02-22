import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

def generator_constraints(Pg, Qg, Pg_min, Pg_max):

    c1 = Pg - Pg_max
    c2 = Pg_min - Pg

    return torch.cat([c1, c2], dim=1)

def phi(c):
    return torch.relu(c) ** 2
