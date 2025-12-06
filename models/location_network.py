import torch
import torch.nn as nn
import torch.distributions as dist


class LocationNetwork(nn.Module):
    """
    Location network that outputs the mean of a Gaussian distribution
    for sampling the next location to attend to.
    """

    def __init__(self, hidden_size, std=0.2):
        """
        Initialize the location network.

        Args:
            hidden_size: Size of hidden state from core network
            std: Standard deviation of the location policy (fixed variance)
        """
        super().__init__()
        self.std = std
        self.location_linear = nn.Linear(hidden_size, 2)  # (x, y) coordinates

    def forward(self, hidden_state):
        """
        Forward pass through the location network.

        Args:
            hidden_state: Hidden state from core network

        Returns:
            Mean of location distribution (x, y)
        """
        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)

        mean = self.location_linear(hidden_state)
        # Clamp to [-1, 1] range
        mean = torch.tanh(mean)
        return mean.squeeze(0) if mean.shape[0] == 1 else mean

    def sample(self, hidden_state):
        """
        Sample a location from the location policy.

        Args:
            hidden_state: Hidden state from core network

        Returns:
            Sampled location (x, y) and log probability
        """
        mean = self.forward(hidden_state)
        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        # Create Gaussian distribution
        loc_dist = dist.Normal(mean, self.std)
        location = loc_dist.sample()
        log_prob = loc_dist.log_prob(location).sum(dim=-1)

        # Clamp location to [-1, 1]
        location = torch.clamp(location, -1.0, 1.0)

        return location.squeeze(0) if location.shape[0] == 1 else location, log_prob

    def log_prob(self, location, hidden_state):
        """
        Compute log probability of a location given hidden state.

        Args:
            location: Location coordinates (x, y)
            hidden_state: Hidden state from core network

        Returns:
            Log probability
        """
        mean = self.forward(hidden_state)
        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)
        if len(location.shape) == 1:
            location = location.unsqueeze(0)

        loc_dist = dist.Normal(mean, self.std)
        log_prob = loc_dist.log_prob(location).sum(dim=-1)
        return log_prob.squeeze(0) if log_prob.shape[0] == 1 else log_prob

