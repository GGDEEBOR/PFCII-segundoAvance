import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim, horizon):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, theta_dim)
        self.fc2 = nn.Linear(theta_dim, theta_dim)
        self.backcast_fc = nn.Linear(theta_dim, input_dim)
        self.forecast_fc = nn.Linear(theta_dim, horizon)

    def forward(self, x):
        theta = torch.relu(self.fc1(x))
        theta = torch.relu(self.fc2(theta))
        backcast = self.backcast_fc(theta)
        forecast = self.forecast_fc(theta)
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_dim=168, horizon=48, num_blocks=3, theta_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, theta_dim, horizon) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        forecast_stack = []
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast
            forecast_stack.append(forecast)
        return torch.sum(torch.stack(forecast_stack), dim=0)