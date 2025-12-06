import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Value network that estimates the expected cumulative reward (baseline)
    for variance reduction in REINFORCE.
    """

    def __init__(self, hidden_size):
        """
        Initialize the value network.

        Args:
            hidden_size: Size of hidden state from core network
        """
        super().__init__()
        self.value_linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state):
        """
        Forward pass through the value network.

        Args:
            hidden_state: Hidden state from core network

        Returns:
            Estimated value (expected cumulative reward)
        """
        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)

        value = self.value_linear(hidden_state)
        return value.squeeze(-1).squeeze(0) if value.shape[0] == 1 else value.squeeze(-1)

