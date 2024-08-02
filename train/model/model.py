
import torch
import torch.nn as nn

class FeatureNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list):
        super(FeatureNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], 1))  # Output a single value for each feature
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CoxNAM(nn.Module):
    def __init__(self, num_features: int, input_dim: int, hidden_units: list):
        super(CoxNAM, self).__init__()
        self.feature_networks = nn.ModuleList([
            FeatureNetwork(input_dim, hidden_units) for _ in range(num_features)
        ])

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        contributions = [network(x[:, i, :]) for i, network in enumerate(self.feature_networks)]
        risk_scores = torch.sum(torch.stack(contributions, dim=1), dim=1)  # Stack along the correct dimension
        return risk_scores
