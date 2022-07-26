from torch import nn
from ireul.common.utils import FactorizedNoisyLinear

class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.layers(x)

class NoisyDQN(nn.Module):

    def __init__(self, n_inputs, n_actions):
        super().__init__()

        self.noisy_fc = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            FactorizedNoisyLinear(128, 128),
            nn.ReLU(),
            FactorizedNoisyLinear(128, n_actions)
        )
    
    def forward(self, x):
        return self.noisy_fc(x)

class DuelingDQN(nn.Module):

    def __init__(self, n_inputs, n_actions):
        super().__init__()
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        features = self.feauture_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean())
        